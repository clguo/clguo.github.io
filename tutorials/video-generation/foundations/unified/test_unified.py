import unittest
import torch
from torch import nn
from scipy.optimize import linear_sum_assignment

from digit_model import DigitUNet
from .processes import cosine_schedule,generate,training_pair


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_reference_rf_time_reversal_matches_fm(self):
        torch.manual_seed(4)
        data=torch.randn(3,1,4,4);noise=torch.randn_like(data);t=torch.rand(3)
        xt,target,_=training_pair(data,noise,t,'flow')
        backwards=(1-t)[:,None,None,None]
        original_xt=backwards*noise+(1-backwards)*data
        drift=noise-data
        self.assertTrue(torch.allclose(xt,original_xt))
        self.assertTrue(torch.equal(target,-drift))
        self.assertTrue(torch.allclose(xt+.1*target,original_xt-.1*drift))

    def test_both_diffusion_samplers_recover_oracle_clean_images(self):
        torch.manual_seed(1)
        clean=torch.rand(3,1,4,4)*2-1
        class Oracle(nn.Module):
            def __init__(self):super().__init__();self.a=cosine_schedule();self.calls=0
            def forward(self,x,t,y):
                self.calls+=1
                a=self.a[(t*1000).round().long()-1][:,None,None,None]
                return (x-a.sqrt()*clean)/(1-a).sqrt()
        for method in ['ddpm','ddim']:
            for count in [1,4,64,1000]:
                oracle=Oracle()
                output=generate(oracle,torch.randn_like(clean),torch.arange(3),method,count)
                self.assertTrue(torch.allclose(output,clean,atol=.003))
                self.assertEqual(oracle.calls,count)

    def test_assignment_direction_counterexample(self):
        noise=torch.tensor([[0.],[10.],[20.]])
        data=torch.tensor([[20.],[0.],[10.]])
        rows,cols=linear_sum_assignment(torch.cdist(noise,data).numpy())
        wrong=noise[cols]
        correct=noise.clone();correct[cols]=noise[rows]
        self.assertEqual((wrong-data).abs().sum().item(),40.)
        self.assertEqual((correct-data).abs().sum().item(),0.)

    def test_model_uses_labels_and_preserves_shape(self):
        torch.manual_seed(42)
        model=DigitUNet(num_classes=10).eval()
        x=torch.randn(2,1,28,28);t=torch.tensor([.2,.8]);y=torch.tensor([1,9])
        output=model(x,t,y)
        self.assertEqual(output.shape,x.shape)
        self.assertFalse(torch.allclose(output,model(x,t,1-y%2)))
        output.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__=='__main__':unittest.main()
