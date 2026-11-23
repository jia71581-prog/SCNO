"""Fit an ordinary-only pressure readout in source-aligned temporal coordinates."""

from argparse import ArgumentParser
from itertools import product
import json
from pathlib import Path
from time import monotonic

import numpy as np
from scipy import fft
from scipy.linalg import cho_factor, cho_solve
from scipy.ndimage import map_coordinates
from scipy.spatial.distance import cdist
import torch
from torch.nn import functional as F

from expanded_train_pressure_archive import ExpandedPressureArchive
from fit_e12_b5 import atomic_npz, integrated_error, sha, write_json
from tianhe_observations import CommittedMarmousi
from train_joint_pressure import PressureArchive


NATIVE_SHAPE = (801, 201, 201)
GRID = np.arange(101, dtype=np.float32) * 20.
Z, X = np.meshgrid(GRID, GRID, indexing="ij")
FREQUENCIES = np.fft.rfftfreq(801, .005)
KEEP = int(np.searchsorted(FREQUENCIES, 28., side="right"))
PACKED_SHAPE = (KEEP, len(GRID), len(GRID))
DESCRIPTOR_SHAPE = (51, 51)


def travel_time(velocity, source, count=24):
    """Integrate slowness along fixed rays; no pressure time marching is involved."""
    sx, sz = source[:2]
    fractions = ((.5 + np.arange(count, dtype=np.float32)) / count)[:, None, None]
    path_z = sz + fractions*(Z - sz)
    path_x = sx + fractions*(X - sx)
    axes = np.stack((path_z, path_x)) / 10.
    sigma = map_coordinates(1. / velocity, axes, order=1, mode="nearest", prefilter=False).mean(axis=0)
    return np.hypot(Z - sz, X - sx)*sigma


def phase_delay(delay, sign):
    omega = (sign*2.*np.pi*FREQUENCIES[:KEEP]).astype(np.float32)
    return np.exp(1j * omega[:, None, None] * delay)


def descriptor(velocity):
    value = np.log(velocity / 2000.)
    return F.avg_pool2d(torch.from_numpy(value)[None, None], 5, stride=4, padding=2,
                        count_include_pad=False)[0,0].numpy().ravel()


def records_for(archive, split):
    for index, (row, _, _) in enumerate(archive.items[split]):
        yield row, archive.arrays(split, index)
    if split == "train":
        for row, folder, velocity in archive._signed.values():
            yield row, archive._new_arrays(folder, velocity)


def make_bank(out, archive, threads):
    info = out/"CACHE.json"
    cache_path = out/"PRESSURE_SPECTRA.npy"
    size = sum(len(value) for value in archive.items.values()) + len(archive._signed)
    if info.is_file():
        data = json.loads(info.read_text())
        if data["binding"] != archive.binding or data["spectra_sha256"] != sha(cache_path):
            raise ValueError("The existing ordinary spectral bank changed")
        return data, np.load(cache_path, mmap_mode="r", allow_pickle=False)
    spectra = np.lib.format.open_memmap(cache_path.with_suffix(".tmp"), mode="w+", dtype=np.complex64,
                                        shape=(size, *PACKED_SHAPE))
    rows, velocities, delay, unseen_energy = [], [], [], []
    started = monotonic()
    for split in ("train", "validation"):
        for row, (pressure, _, velocity) in records_for(archive, split):
            index = len(rows)
            source = archive.release.sources[row["shot_id"]]
            field = np.ascontiguousarray(pressure[:,::2,::2] / archive.release.rms, dtype=np.float32)
            spectrum = fft.rfft(field, axis=0, norm="ortho", workers=threads)
            spectrum[1:] *= np.sqrt(np.float32(2.))
            if not np.isfinite(spectrum).all():
                raise ValueError("Nonfinite archived physical pressure")
            spectra[index] = spectrum[:KEEP]
            unseen_energy.append(float(np.square(np.abs(spectrum[KEEP:]), dtype=np.float64).sum()))
            velocities.append(descriptor(velocity))
            delay.append(travel_time(velocity, source))
            rows.append({k:row[k] for k in ("family", "model_id", "parent_group_id", "shot_id")} | {"split": split})
            if (index+1) % 12 == 0:
                write_json(out/"STATUS.json", dict(status="CACHING", ready=index+1, shots=size,
                                                    elapsed_s=monotonic()-started))
    if len(rows) != size or {r["parent_group_id"] for r in rows if r["split"]=="train"} & archive.parents["validation"]:
        raise ValueError("The ready pressure archives must stay disjoint by ordinary medium")
    spectra.flush()
    del spectra
    cache_path.with_suffix(".tmp").replace(cache_path)
    atomic_npz(out/"GEOMETRY.npz", descriptors=np.stack(velocities), delays=np.stack(delay),
               unseen_energy=np.asarray(unseen_energy))
    d = dict(binding=archive.binding, rows=rows, spectra_sha256=sha(cache_path),
             descriptor_sha256=sha(out/"GEOMETRY.npz"), train=len(rows)-12, validation=12,
             numerical_wave_propagations=0, exact_marmousi_used=False)
    write_json(info, d)
    return d, np.load(cache_path, mmap_mode="r", allow_pickle=False)


def gram_matrix(fields, times, use_alignment, n_train, tail):
    """Use Parseval energies to rank inter-model interpolation without rendering fields."""
    result = np.zeros((len(fields), len(fields)), dtype=np.float64)
    for start in range(0, KEEP, 4):
        indices = slice(start, min(start+4, KEEP))
        block = np.ascontiguousarray(fields[:, indices])
        if use_alignment:
            angles = (2.*np.pi*FREQUENCIES[indices]).astype(np.float32)[None,:,None,None]
            block *= np.exp(1j * angles * times[:,None])
        real = block.view(np.float32).reshape(len(fields), -1)
        result += real @ real.T
    total = np.diag(result)[n_train:].copy() + tail
    return (result[:n_train,:n_train], result[n_train:,:n_train], total), total


def mixtures(descriptors, source_x, n_train):
    depth = np.linspace(0., 2., DESCRIPTOR_SHAPE[0], dtype=np.float64)[:,None]
    depth_weight = np.sqrt(np.broadcast_to(.25+np.exp(-depth/.55), DESCRIPTOR_SHAPE).ravel())
    geometry = (descriptors, descriptors*depth_weight,
                np.diff(descriptors.reshape(-1,*DESCRIPTOR_SHAPE), axis=1).reshape(len(descriptors), -1))
    distances = [cdist(value, value[:n_train], "sqeuclidean") for value in geometry]
    lengths = [np.median(d[:n_train][d[:n_train] > 0.]) for d in distances]
    source_delta = np.square(source_x[:,None]-source_x[None,:n_train])
    for depth_name, mix in (("full", (1.,0.,0.)), ("shallow", (.3,.7,0.)), ("contrast", (.5,.2,.3))):
        for bandwidth, source_width in product((.25,.5,1.,2.), (55.,105.,200.,350.)):
            similarity = sum(w*np.exp(-.5*d/(bandwidth**2*l)) for w,d,l in zip(mix,distances,lengths))
            k = similarity*np.exp(-.5*source_delta/source_width**2)
            sigma = np.diag(k[:n_train]).mean()
            for ridge in (.001,.02,.1):
                system=k[:n_train].copy()
                system.flat[::n_train+1] += ridge*sigma
                weight=cho_solve(cho_factor(system, lower=True), k[n_train:].T).T
                mass = weight.sum(1, keepdims=True)
                if np.any(mass <= .2):
                    continue
                yield dict(geometry=depth_name, medium_bandwidth=bandwidth, source_width_m=source_width,
                           ridge=ridge), weight/mass


def spectral_predictions(fields, weights, delays, query_delays, aligned):
    output = np.zeros((len(weights), *PACKED_SHAPE), dtype=np.complex64)
    coefficients = weights.astype(np.float32)
    for start in range(0, KEEP, 4):
        sl = slice(start, min(start+4, KEEP))
        block = fields[:,sl]
        if aligned:
            frequencies = (2.*np.pi*FREQUENCIES[sl]).astype(np.float32)[None,:,None,None]
            block = block*np.exp(1j*frequencies*delays[:,None])
        predicted = coefficients @ block.reshape(len(block),-1)
        output[:,sl] = predicted.reshape(len(weights),-1,*X.shape)
    output[:,1:] /= np.sqrt(np.float32(2.))
    if aligned:
        for values, delay in zip(output, query_delays, strict=True):
            values *= phase_delay(delay, -1.)
    return output


def score_native_pressure(archive, predicted_spectra):
    total, late, deep = (np.zeros(2) for _ in range(3))
    shots = []
    for index, (row, _, _) in enumerate(archive.items["validation"]):
        reference, _, _ = archive.arrays("validation", index)
        coarse = fft.irfft(predicted_spectra[index], n=801, axis=0, norm="ortho")
        per_shot = np.zeros(2)
        for first in range(0,801,32):
            slots = slice(first,min(801,first+32))
            predicted = F.interpolate(torch.from_numpy(coarse[slots,None]), size=(201,201),
                                      mode="bicubic", align_corners=True)[:,0].numpy()
            predicted[:,0] = 0.
            if first == 0:
                predicted[:3] = 0.
            target = reference[slots].astype(np.float64)/archive.release.rms
            error = np.square(predicted-target)
            energy = np.square(target)
            active = (np.arange(slots.start,slots.stop) >= 200) & (np.arange(slots.start,slots.stop) < 401)
            per_shot += [error.sum(),energy.sum()]
            late += [error[active].sum(),energy[active].sum()]
            deep += [error[:,60:181].sum(),energy[:,60:181].sum()]
        total += per_shot
        shots.append(dict(model_id=row["model_id"], shot_id=row["shot_id"], family=row["family"],
                          full_pressure_relative_l2=float(np.sqrt(per_shot[0]/per_shot[1]))))
    return {k:float(np.sqrt(v[0]/v[1])) for k,v in (("full_pressure_relative_l2",total),
            ("pressure_1_to_2s_relative_l2",late),("pressure_600_to_1800m_relative_l2",deep))} | {"shots":shots}


def native_scores(archive, fields, weights, times, n_train, aligned):
    spectra = spectral_predictions(fields[:n_train],weights,times[:n_train],times[n_train:],aligned)
    return score_native_pressure(archive,spectra)


def run(args):
    torch.set_num_threads(args.threads)
    release = CommittedMarmousi(args.dataset)
    old = PressureArchive(args.pressure_bank, release)
    archive = ExpandedPressureArchive(old, args.expanded_bank, bank_a=args.bank_a, bank_f=args.bank_f)
    args.output.mkdir(parents=True, exist_ok=True)
    receipt, fields = make_bank(args.output, archive, args.threads)
    geometry_file = args.output/"GEOMETRY.npz"
    if receipt["descriptor_sha256"] != sha(geometry_file):
        raise ValueError("The cached training velocity descriptors changed")
    with np.load(geometry_file,allow_pickle=False) as data:
        descriptors, times, tail = [data[key] for key in ("descriptors","delays","unseen_energy")]
    n_train = receipt["train"]
    shot_positions = np.asarray([release.sources[row["shot_id"],0] for row in receipt["rows"]])
    results = {}
    for label, aligned in (("absolute_time", False), ("ray_aligned_time", True)):
        gram, energy = gram_matrix(fields, times, aligned, n_train, tail[n_train:])
        candidates = ((float(np.sqrt(integrated_error(w,gram).sum()/energy.sum())), p, w)
                      for p,w in mixtures(descriptors,shot_positions,n_train))
        score, params, selected = min(candidates, key=lambda candidate: candidate[0])
        result = native_scores(archive, fields, selected, times, n_train, aligned)
        results[label] = dict(parameters=params, validated_coarse_relative_l2=score, **result)
        print(json.dumps({k:v for k,v in results[label].items() if k!="shots"}),flush=True)
    write_json(args.output/"RESULT.json", dict(status="COMPLETE", reference=receipt["binding"],
               train_shots=n_train, validation_shots=12, time_samples=801, grid_spacing_m=10.,
               exact_marmousi_used=False, numerical_wave_propagations=0, results=results))
    write_json(args.output/"STATUS.json", dict(status="COMPLETE", filled=len(fields)))


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("dataset","pressure-bank","expanded-bank","bank-a","bank-f","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--threads", type=int,default=8)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
