
import unittest
import numpy as np
import pyfluids


class FailureNrhyd(unittest.TestCase):
    def setUp(self):
        self.S = pyfluids.FluidState(fluid='nrhyd')
    def test_goodc2p(self):
        self.S.primitive = [1.0, 1.0, 0.0, 0.0, 0.0]
        U = self.S.conserved()
        self.assertTrue(np.allclose(U, [1.0, 2.5, 0.0, 0.0, 0.0]))
    def test_negative_density(self):
        with self.assertRaises(pyfluids.fluids.NegativeDensityCons):
            self.S.from_conserved(np.array([-1.0, +1.0, 0.0, 0.0, 0.0]))
    def test_negative_energy(self):
        with self.assertRaises(pyfluids.fluids.NegativeEnergy):
            self.S.from_conserved(np.array([+1.0, -1.0, 0.0, 0.0, 0.0]))
    def test_bad_c2p(self):
        with self.assertRaises(pyfluids.fluids.NegativePressure):
            self.S.from_conserved(np.array([+1.0, 0.5, 2.0, 0.0, 0.0]))
    def test_negative_pressure(self):
        self.S.primitive = [1.0, -1.0, 0.0, 0.0, 0.0]
        with self.assertRaises(pyfluids.fluids.NegativePressure):
            U = self.S.conserved()


class FailureSrhyd(unittest.TestCase):
    def setUp(self):
        self.S = pyfluids.FluidState(fluid='srhyd')
    def test_goodc2p(self):
        self.S.primitive = [1.0, 1.0, 0.0, 0.0, 0.0]
        U = self.S.conserved()
        self.assertTrue(np.allclose(U, [1.0, 2.5, 0.0, 0.0, 0.0]))
    def test_negative_density(self):
        with self.assertRaises(pyfluids.fluids.NegativeDensityCons):
            self.S.from_conserved(np.array([-1.0, +1.0, 0.0, 0.0, 0.0]))
    def test_negative_energy(self):
        with self.assertRaises(pyfluids.fluids.NegativeEnergy):
            self.S.from_conserved(np.array([+1.0, -1.0, 0.0, 0.0, 0.0]))
    def test_bad_c2p(self):
        with self.assertRaises(pyfluids.fluids.ConsToPrimMaxIteration):
            self.S.from_conserved(np.array([+1.0, 0.5, 2.0, 0.0, 0.0]))
    def test_negative_pressure(self):
        self.S.primitive = [1.0, -1.0, 0.0, 0.0, 0.0]
        with self.assertRaises(pyfluids.fluids.NegativePressure):
            U = self.S.conserved()
    def test_superluminal(self):
        self.S.primitive = [1.0, 1.0, 1.1, 0.0, 0.0]
        with self.assertRaises(pyfluids.fluids.SuperluminalVelocity):
            U = self.S.conserved()
    def test_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            A = self.S.jacobian()


class FailureFluidStateVector(unittest.TestCase):
    def setUp(self):
        self.fluid = pyfluids.FluidStateVector([4,4], fluid='nrhyd')
    def test_failmask_p2c(self):
        self.fluid.primitive = [1.0, 1.0, 0.0, 0.0, 0.0]
        self.fluid.states[2,2].primitive = [-1.0, 1.0, 0.0, 0.0, 0.0]
        self.fluid.failmask = 0
        U = self.fluid.conserved()
        self.assertNotEqual(self.fluid.failmask[2,2], 0)
        self.fluid.failmask[2,2] = 0
        self.assertTrue((self.fluid.failmask == 0).all())
    def test_failmask_c2p(self):
        self.fluid.primitive = [1.0, 1.0, 0.0, 0.0, 0.0]
        U = self.fluid.conserved()
        U[2,2,1] = -1.0
        self.fluid.from_conserved(U)
        self.assertNotEqual(self.fluid.failmask[2,2], 0)
        

if __name__ == '__main__':
    unittest.main()
