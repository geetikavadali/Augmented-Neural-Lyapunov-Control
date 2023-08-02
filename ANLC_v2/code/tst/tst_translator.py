#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 19:02:40 2023
@authors: Andrea Peruffo


"""

import numpy as np
import torch
import sys
import os
import dreal as dr
import unittest

sys.path.append(os.getcwd() + '/..')

from utilities import translator
from utilities.nn import Net
from configuration import config_2d_template as config_file
from utilities.models import test_f
from utilities.Functions import CheckLyapunov
from utilities.from_dreal_to_np import sub as dr2np


class TranslatorTest(unittest.TestCase):
    def setUp(self) -> None:

        torch.set_default_dtype(torch.float64)
        self.seed_ = 777.
        self.params = config_file.get_params()
        self.params['lyap_hid1'] = 2
        self.params['lyap_hid2'] = 2
        self.params['size_layers'] = [2, 2, 1]

        self.model = Net(self.params, self.seed_)

        self.vars_ = [dr.Variable('x1'), dr.Variable('x2')]

    def identity_lyap_easy_ctrl(self):
        # set weights as V = x1**2 + x2**2
        self.model.layers[0].weight.data = torch.eye(2)
        self.model.layers[1].weight.data = torch.eye(2)
        self.model.layers[2].weight.data = torch.atleast_2d(torch.tensor([1., 1.]))
        # set control weight so that u = 0.2 * x2
        self.model.ctrl_layers[0].weight.data = torch.atleast_2d(torch.tensor([0., 0.2]))
        self.model.ctrl_layers[1].weight.data = torch.atleast_2d(torch.tensor([1.]))
        self.model.ctrl_layers[2].weight.data = torch.atleast_2d(torch.tensor([1.]))

        # now we should have V = x1**2 + x2**2 and f(x) = [ -0.5*x1, -0.8*x2 ]

    def medium_network(self):
        self.seed_ = 777.
        self.params = config_file.get_params()
        self.params['lyap_hid1'] = 10
        self.params['lyap_hid2'] = 10

        self.params['size_layers'] = [10, 10, 1]

        self.model = Net(self.params, self.seed_)

    def complex_network(self):
        self.seed_ = 777.
        self.params = config_file.get_params()

        self.model = Net(self.params, self.seed_)

    def test_check_V_and_Vdot_are_what_is_supposed_to_be(self):

        self.identity_lyap_easy_ctrl()

        # dreal section -- symbolic evaluation
        x_star = [0., 0.]
        f_symb = test_f.f_symb

        u, self.V, self.f_out_sym = translator(self.params, self.vars_, self.model, x_star, f_symb)
        # V = x1**2 + x2**2 and f(x) = [ -0.5*x1, -0.8*x2 ]
        # then dV = [ 2*x1, 2*x2 ]
        # Vdot = -x1 ** 2 - 1.6 * x2 ** 2
        _, vdot = CheckLyapunov(self.vars_, self.f_out_sym, self.V, 1., 10., 1e-6, 0.)

        # almost equal because numpy / dReal mess with numbers
        self.assertEqual(self.V, self.vars_[0] ** 2 + self.vars_[1] ** 2)

        # f1(x) should be -0.5*x1, but numpy /dreal mess with number so check that
        # f1(x) + 0.5*x1 is almost zero
        self.assertAlmostEqual(self.f_out_sym[0] + 0.5 * self.vars_[0], 0.)

        # f2(x) should be -0.8*x2, but numpy /dreal mess with number so check that
        # f2(x) + 0.8*x2 is almost zero
        self.assertAlmostEqual(self.f_out_sym[1] + 0.8 * self.vars_[1], 0.)

        difference = -self.vars_[0] ** 2 - 1.6 * self.vars_[1] ** 2 - vdot
        self.assertAlmostEqual(difference, 0.)

    def test_evaluate_V_and_Vdot_are_what_is_supposed_to_be(self):

        self.identity_lyap_easy_ctrl()

        # now we should have V = x1**2 + x2**2 and f(x) = [ -0.5*x1, -0.8*x2 ]

        # dreal section -- symbolic evaluation
        x_star = [0., 0.]
        f_symb = test_f.f_symb

        dr_u, dr_V, dr_fx = translator(self.params, self.vars_, self.model, x_star, f_symb)
        # V = x1**2 + x2**2 and f(x) = [ -0.5*x1, -0.8*x2 ]
        # then dV = [ 2*x1, 2*x2 ]
        # Vdot = -x1 ** 2 - 1.6 * x2 ** 2
        _, dr_vdot = CheckLyapunov(self.vars_, dr_fx, dr_V, 1., 10., 1e-6, 0.)

        # numerivcal evaluation
        n_points = 10
        x = torch.randn((n_points, len(self.vars_)))
        x_star = [0., 0.]

        # numerical
        torchV, torchVdot, _, torchu = self.model.forward(x, test_f.f_torch, self.params, x_star)
        torch_fx = test_f.f_torch(x, torchu, self.params, x_star)

        eval_v = np.zeros(n_points)
        eval_vdot = np.zeros(n_points)
        eval_u = np.zeros(n_points)
        eval_fx = np.zeros(n_points)

        for n in range(n_points):
            replacements = {self.vars_[i]: float(x[n, i]) for i in range(len(self.vars_))}
            # cannot find a better way to do this....
            eval_v[n] = float(str(dr_V.Substitute(replacements)))
            eval_u[n] = float(str(dr_u[0].Substitute(replacements)))
            eval_fx[n] = float(str(dr_fx[0].Substitute(replacements)))
            eval_vdot[n] = float(str(dr_vdot.Substitute(replacements)))

        # prepare for comparison
        tv, tvdot = torchV.detach().numpy(), torchVdot.detach().numpy()
        tu, tfx = torchu[:, 0].detach().numpy(), torch_fx[:, 0].detach().numpy()
        self.assertTrue((abs(eval_v - eval_v) < 1e-6).all())
        self.assertTrue((abs(eval_vdot - tvdot) < 1e-6).all())
        self.assertTrue((abs(eval_u - tu) < 1e-6).all())
        self.assertTrue((abs(eval_fx - tfx) < 1e-6).all())

    def test_complex_check_V_and_Vdot_are_what_is_supposed_to_be(self):
        # set weights as V = x2**2 + (x1 + 7*x2) **2 = x1**2 + 14*x1*x2 + 50*x2**2
        self.model.layers[0].weight.data = torch.eye(2)
        self.model.layers[1].weight.data = torch.tensor([[1., 7.], [0., 1.]])
        self.model.layers[2].weight.data = torch.atleast_2d(torch.tensor([1., 1.]))
        # set control weight so that u = 0.2 * x2
        self.model.ctrl_layers[0].weight.data = torch.atleast_2d(torch.tensor([0., 0.2]))
        self.model.ctrl_layers[1].weight.data = torch.atleast_2d(torch.tensor([1.]))
        self.model.ctrl_layers[2].weight.data = torch.atleast_2d(torch.tensor([1.]))

        # now we should have V = x1**2 + x2**2 + 7*x1*x2 and f(x) = [ -0.5*x1, -0.8*x2 ]

        # dreal section -- symbolic evaluation
        x_star = [0., 0.]
        f_symb = test_f.f_symb

        u, self.V, self.f_out_sym = translator(self.params, self.vars_, self.model, x_star, f_symb)
        # V = x1**2 + 14*x1*x2 + 50*x2**2 and f(x) = [ -0.5*x1, -0.8*x2 ]
        # then dV = [ 2*x1 + 14*x2, 100*x2 + 14*x1 ]
        # Vdot = -x1 ** 2 - 80 * x2 ** 2  - 18.2 * x1*x2
        _, vdot = CheckLyapunov(self.vars_, self.f_out_sym, self.V, 1., 10., 1e-6, 0.)

        # check v - desired_v == 0
        desired_v = self.vars_[0] ** 2 + 14. * self.vars_[0] * self.vars_[1] + 50. * self.vars_[1] ** 2
        self.assertEqual(0., (desired_v - self.V).Expand())

        # f1(x) should be -0.5*x1, but numpy /dreal mess with number so check that
        # f1(x) + 0.5*x1 is almost zero
        self.assertAlmostEqual(self.f_out_sym[0] + 0.5 * self.vars_[0], 0.)

        # f2(x) should be -0.8*x2, but numpy /dreal mess with number so check that
        # f2(x) + 0.8*x2 is almost zero
        self.assertAlmostEqual(self.f_out_sym[1] + 0.8 * self.vars_[1], 0.)

        desired_vdot = -self.vars_[0] ** 2 - 80 * self.vars_[1] ** 2 - 18.2 * self.vars_[0] * self.vars_[1]
        difference = (desired_vdot - vdot).Expand()
        # actually, the result is epsilon * x2; check that epsilon is small
        # find where the difference is greater than 1
        box = dr.CheckSatisfiability(difference.Expand() > 1, 1e-6)
        mid = []
        for i in range(len(self.vars_)):
            mid.append(box[i].mid())

        # check that epsilon is small, i.e. 1/epsilon > 1e100
        self.assertTrue(all([abs(mid[i]) > 1e100 for i in range(len(self.vars_))]))

    def test_evaluate_V_and_Vdot(self):

        self.complex_network()

        n_points = 10000
        # random in [-10, 10]
        x = 20. * torch.rand((n_points, len(self.vars_))) + -10.
        x_star = [0., 0.]

        # numerical
        torchV, torchVdot, _, torchu = self.model.forward(x, test_f.f_torch, self.params, x_star)
        torch_fx = test_f.f_torch(x, torchu, self.params, x_star)

        # dreal section -- symbolic evaluation
        dr_u, dr_V, dr_fx = translator(self.params, self.vars_, self.model, x_star, test_f.f_symb)
        _, dr_vdot = CheckLyapunov(self.vars_, dr_fx, dr_V, 1., 10., 1e-6, 0.)

        eval_v = np.zeros(n_points)
        eval_vdot = np.zeros(n_points)
        eval_u = np.zeros(n_points)
        eval_fx = np.zeros(n_points)

        for n in range(n_points):
            replacements = {self.vars_[i]: float(x[n, i]) for i in range(len(self.vars_))}
            # cannot find a better way to do this....
            eval_v[n] = dr_V.Substitute(replacements).Evaluate()
            eval_u[n] = dr_u[0].Substitute(replacements).Evaluate()
            eval_fx[n] = dr_fx[0].Substitute(replacements).Evaluate()
            eval_vdot[n] = dr_vdot.Substitute(replacements).Evaluate()

        # prepare for comparison
        tv, tvdot = torchV.detach().numpy(), torchVdot.detach().numpy()
        tu, tfx = torchu[:, 0].detach().numpy(), torch_fx[:, 0].detach().numpy()

        print(f'Max Difference in Vdot: {max(abs(eval_vdot-tvdot))}')

        self.assertTrue((abs(eval_v - eval_v) < 1e-6).all())
        self.assertTrue((abs(eval_vdot - tvdot) < 1e-6).all())
        self.assertTrue((abs(eval_u - tu) < 1e-6).all())
        self.assertTrue((abs(eval_fx - tfx) < 1e-6).all())

    def test_augmentedfalsifier(self):

        self.complex_network()

        n_points = 100
        bound = 1.

        X1 = np.linspace(-bound, bound, n_points)
        X2 = np.linspace(-bound, bound, n_points)
        x1, x2 = np.meshgrid(X1, X2)

        x = torch.tensor(np.vstack([x1.ravel(), x2.ravel()])).T
        x_star = [0., 0.]

        # numerical
        torchV, torchVdot, _, torchu = self.model.forward(x, test_f.f_torch, self.params, x_star)
        torch_fx = test_f.f_torch(x, torchu, self.params, x_star)

        # dreal section -- symbolic evaluation
        dr_u, dr_V, dr_fx = translator(self.params, self.vars_, self.model, x_star, test_f.f_symb)
        _, dr_vdot = CheckLyapunov(self.vars_, dr_fx, dr_V, 1., 10., 1e-6, 0.)

        # augmented falsifier
        dV_sub = dr2np(dr_vdot.to_string())  # substitute dreal functions
        dV_eval = eval(dV_sub)
        # put in a 2d format
        dV_eval = torch.tensor(dV_eval.reshape(-1))

        # comparison
        print(f'Max difference: {max(abs(dV_eval-torchVdot))}')

        self.assertTrue(max(abs(dV_eval-torchVdot)) < 1e-6)




