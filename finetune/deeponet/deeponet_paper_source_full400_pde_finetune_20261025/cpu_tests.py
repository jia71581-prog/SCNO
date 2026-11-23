import copy,tempfile,unittest
from pathlib import Path
import torch
import finetune as f
from model import RepairDeepONet,tiny_architecture
class Tests(unittest.TestCase):
 def test_mixed_loss_exact(self):
  torch.manual_seed(3);truth=torch.randn(1,400,3,3);prediction=torch.randn_like(truth)
  stage={'training':{'data_window_boundaries':[0,100,200,300,400],'receiver_indices':[[1,1]]}}
  sup=f.mixed_supervision(truth,stage);error=(prediction-truth).square();actual=(error*sup['frame_weights'][:,:,None,None]).mean((1,2,3))/sup['energy']
  expected=.5*error.mean((1,2,3))/truth.square().mean((1,2,3))+.5*(error.mean((-2,-1))/truth.square().mean((-2,-1))).mean(1)
  self.assertTrue(torch.allclose(actual,expected,rtol=1e-6,atol=1e-6))
 def test_zero_frame_rejected(self):
  truth=torch.ones(1,400,3,3);truth[:,9]=0
  with self.assertRaises(AssertionError):f.mixed_supervision(truth,{'training':{'data_window_boundaries':[0,100,200,300,400],'receiver_indices':[[1,1]]}})
 def test_chunk_fallback_finite(self):
  c=f.CONFIG['chunk_candidates'];self.assertEqual([f.next_chunk(x,c) for x in c],[1024,256,128,None])
 def test_budget_and_cursor(self):
  p=f.initial_progress();f.validate_progress(p,f.CONFIG)
  p.update(updates=2000,committed_fit_record_exposures=2000);f.validate_progress(p,f.CONFIG)
  p['updates']=2001
  with self.assertRaises(AssertionError):f.validate_progress(p,f.CONFIG)
 def test_complete_cpu_restore_next_step(self):
  torch.manual_seed(9);cfg=copy.deepcopy(f.CONFIG);cfg['architecture']=tiny_architecture();net=RepairDeepONet(cfg['architecture']);opt=f.make_optimizer(net,cfg)
  self.assertEqual(len(opt.param_groups),4);self.assertFalse(opt.state);self.assertTrue(all(p.requires_grad for p in net.parameters()))
  x=torch.randn(1,16,17,17);q=torch.rand(13,3);y=torch.randn(1,13)
  def step(n,o):o.zero_grad();(n(x,q)-y).square().mean().backward();o.step()
  step(net,opt);progress=f.initial_progress();progress.update(updates=1,committed_fit_record_exposures=1)
  saved=dict(schema=f.SCHEMA,config=cfg,progress=progress,model=f.old.cpu_tree(net.state_dict()),optimizer=f.old.cpu_tree(opt.state_dict()),rng=f.old.get_rng('cpu'))
  with tempfile.TemporaryDirectory(dir=f.RUN) as tmp:
   path=Path(tmp)/'latest.pt';f.atomic_checkpoint(saved,path);loaded=torch.load(path,map_location='cpu',weights_only=False);self.assertTrue(f.old.equal_tree(loaded,saved))
  other,opt2=f.restore_objects(loaded,'cpu');step(net,opt);step(other,opt2)
  self.assertTrue(f.old.equal_tree(net.state_dict(),other.state_dict()));self.assertTrue(f.old.equal_tree(opt.state_dict(),opt2.state_dict()));self.assertFalse(torch.cuda.is_initialized())
 def test_parameter_change_reports_exact_groups(self):
  net=RepairDeepONet(tiny_architecture());before=f.old.cpu_tree(net.state_dict());after=copy.deepcopy(before);after['branch.4.weight']+=.1
  change=f.parameter_change(before,after,f.inherited.parameter_roles(net));self.assertGreater(change['branch_head']['relative_l2'],0);self.assertEqual(change['trunk_head']['relative_l2'],0)
if __name__=='__main__':torch.set_num_threads(2);unittest.main()
