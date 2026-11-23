import copy,unittest
from unittest.mock import patch
from pathlib import Path
import torch
import train
from model import RepairDeepONet,tiny_architecture
import metrics
class JointTests(unittest.TestCase):
 def test_budget_history_and_tails(self):
  p=train.initial_progress();self.assertEqual((p['updates'],p['exposures'],p['offset']),(26,296,296));batches=[]
  while train.next_ids(p):
   ids=train.next_ids(p);batches.append(ids);train.advance(p,ids,train.CONFIG['seed'])
  self.assertEqual((p['updates'],p['segment_updates'],p['exposures'],p['epochs']),(421,395,5000,10))
  self.assertEqual(sum(map(len,batches)),4704);self.assertEqual(sum(len(x)==8 for x in batches),9)
  self.assertEqual(len(set(sum(batches[:17],[]))),204)
 def test_joint_active_fresh_optimizer(self):
  net=RepairDeepONet(tiny_architecture());opt=train.make_optimizer(net,'joint',train.CONFIG)
  self.assertTrue(all(p.requires_grad for p in net.parameters()));self.assertEqual(set(g['label'].split('_')[0] for g in opt.param_groups),{'trunk','branch'});self.assertEqual(len(opt.state),0)
  for epoch in range(10):self.assertEqual(train.phase_for_epoch(epoch),'joint')
 def test_four_role_partition_and_bias(self):
  net=RepairDeepONet(tiny_architecture());cfg=copy.deepcopy(train.CONFIG);cfg['learning_rates']={'branch_body':1e-4,'branch_head':6e-6,'trunk_body':5e-5,'trunk_head':3e-6}
  roles=train.parameter_roles(net);self.assertEqual(roles['output_bias'],'trunk_body')
  self.assertEqual(roles['branch.4.weight'],'branch_head');self.assertEqual(roles['trunk.output.weight'],'trunk_head')
  opt=train.make_optimizer(net,'joint',cfg);self.assertEqual(len(opt.param_groups),4)
  self.assertEqual(sum(len(g['params']) for g in opt.param_groups),len(list(net.parameters())))
  self.assertEqual({g['label']:g['lr'] for g in opt.param_groups},cfg['learning_rates'])
 def test_new_phase_warmup_is_record_clock(self):
  p=train.initial_progress();self.assertEqual(train.warmup_factor(p,12),.06)
  p['exposures']=484;self.assertEqual(train.warmup_factor(p,12),1.)
 def test_actual_branch_not_fixed_codes_and_finite_both_gradients(self):
  net=RepairDeepONet(tiny_architecture());train.make_optimizer(net,'joint',train.CONFIG)
  data={'inputs':torch.randn(2,16,17,17),'truth':torch.randn(2,9)};q=torch.rand(9,3)
  def actual(network,kind,data,*args):
   self.assertIs(network,net);loss=(network(data['inputs'],q)-data['truth']).square().mean();loss.backward();return {'loss':float(loss.detach())}
  with patch.object(train.previous.base,'supervision',return_value={}),patch.object(train.previous,'backward_complete',side_effect=actual):
   result=train.one_backward(net,data,None,None,{},train.CONFIG,torch.full((2,8),float('nan')),'joint',0)
  for group in ('trunk','branch'):
   grad=[p.grad for n,p in net.named_parameters() if (n.startswith('trunk.') or n=='output_bias')==(group=='trunk')]
   self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grad));self.assertGreater(sum(float(g.abs().sum()) for g in grad),0.)
  self.assertEqual(result['coefficient_weight'],0.)
 def test_framewise_zero_mask_and_macro(self):
  truth=torch.tensor([0.,1.,100.]).reshape(1,3,1,1);pred=torch.tensor([5.,0.,100.]).reshape_as(truth)
  r=metrics.framewise_relative_l2_records(pred,truth)[0]
  self.assertEqual(r['zero_reference_frames'],1);self.assertEqual(r['valid_reference_frames'],2)
  self.assertEqual(r['mean_relative_l2'],.5);self.assertEqual(r['per_frame_relative_l2'],[None,1.,0.])
  truth2=torch.zeros_like(truth);self.assertIsNone(metrics.framewise_relative_l2_records(pred,truth2)[0]['mean_relative_l2'])
 def test_joint_checkpoint_restore_next_update(self):
  torch.manual_seed(5);net=RepairDeepONet(tiny_architecture());opt=train.make_optimizer(net,'joint',train.CONFIG)
  x=torch.randn(2,16,17,17);q=torch.rand(9,3);target=torch.randn(2,9)
  def step(n,o):o.zero_grad();(n(x,q)-target).square().mean().backward();o.step()
  step(net,opt);other=RepairDeepONet(tiny_architecture());other.load_state_dict(copy.deepcopy(net.state_dict()));opt2=train.make_optimizer(other,'joint',train.CONFIG);opt2.load_state_dict(copy.deepcopy(opt.state_dict()));step(net,opt);step(other,opt2)
  self.assertTrue(train.old.equal_tree(net.state_dict(),other.state_dict()));self.assertTrue(train.old.equal_tree(opt.state_dict(),opt2.state_dict()))
if __name__=='__main__':torch.set_num_threads(2);unittest.main()
