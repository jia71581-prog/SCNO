import unittest,sys,copy
from pathlib import Path
RUN=Path(__file__).resolve().parent
sys.path.insert(0,str(RUN))
import torch

class Tests(unittest.TestCase):
 def modules(self):
  self.assertTrue((RUN/'model.py').exists(),'model implementation absent')
  self.assertTrue((RUN/'metrics.py').exists(),'metrics implementation absent')
  import model,metrics
  return model,metrics
 def test_parameter_budget(self):
  m,_=self.modules();net=m.RepairDeepONet()
  n=sum(p.numel() for p in net.parameters())
  self.assertGreaterEqual(n,14578039);self.assertLess(abs(n/14578039-1),.01)
 def test_metrics_exact_zero_gain(self):
  _,m=self.modules();truth=torch.arange(1,25,dtype=torch.float32).reshape(2,3,2,2)*1e-9
  self.assertEqual(m.relative_l2_records(truth,truth)[0]['relative_l2'],0.)
  self.assertAlmostEqual(m.relative_l2_records(truth*0,truth)[0]['relative_l2'],1.)
  self.assertAlmostEqual(m.relative_l2_records(truth*2,truth)[0]['relative_l2'],1.,places=6)
  self.assertIsNone(m.relative_l2_records(truth,truth*0)[0]['relative_l2'])
 def test_macro_is_not_pooled(self):
  _,m=self.modules();t=torch.tensor([1.,100.]).reshape(2,1,1,1);p=torch.tensor([0.,100.]).reshape_as(t)
  self.assertEqual(m.aggregate(m.relative_l2_records(p,t))['macro_relative_l2'],.5)
 def test_forecast_excludes_first8(self):
  _,m=self.modules();t=torch.ones(1,400,3,3)*1e-9;p=t.clone();p[:,:8]=0
  a=m.field_metrics(p,t,receivers=[[1,0],[1,1],[1,2]])
  self.assertEqual(a['forecast392'][0]['relative_l2'],0.)
  self.assertAlmostEqual(a['full400'][0]['relative_l2'],(8/400)**.5)
 def test_qr_preserves_lookup_pressure(self):
  m,_=self.modules();torch.manual_seed(7);net=m.RepairDeepONet(m.tiny_architecture());q=torch.rand(512,3);c=torch.randn(6,8)
  with torch.no_grad():before=net.decode_trunk(c,q)
  c2,proof=m.qr_condition(net,c,q)
  with torch.no_grad():after=net.decode_trunk(c2,q)
  self.assertLess(float((after-before).norm()/before.norm()),1e-4)
  self.assertLess(proof['orthogonality_relative_error'],1e-4)
 def test_dense_and_chunk_gradient(self):
  m,_=self.modules();torch.manual_seed(12);net=m.RepairDeepONet(m.tiny_architecture());other=copy.deepcopy(net)
  x=torch.randn(2,16,17,17)*.1;q=torch.rand(37,3);target=torch.randn(2,37)*.01
  ((net(x,q)-target)**2).mean().backward()
  b=other.encode_branch(x);leaf=b.detach().requires_grad_(True)
  for a in range(0,37,7):((other.decode_trunk(leaf,q[a:a+7])-target[:,a:a+7])**2).sum().div(target.numel()).backward()
  b.backward(leaf.grad)
  for p,z in zip(net.parameters(),other.parameters()):
   self.assertIsNotNone(p.grad);self.assertIsNotNone(z.grad)
   self.assertTrue(torch.allclose(p.grad,z.grad,rtol=1e-4,atol=1e-7))
 def test_source_and_known8_connected(self):
  m,_=self.modules();net=m.RepairDeepONet(m.tiny_architecture());x=(torch.randn(2,16,17,17)*.1).requires_grad_();net(x,torch.rand(11,3)).sum().backward()
  self.assertGreater(float(x.grad[:,1:3].abs().sum()),0.)
  self.assertGreater(float(x.grad[:,8:].abs().sum()),0.)

if __name__=='__main__':
 torch.set_num_threads(2);unittest.main()
