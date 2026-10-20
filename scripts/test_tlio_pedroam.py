"""Numerical and input-contract checks for the PedROAM TLIO adapter."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from tlio_pedroam import prepare_body, FILTER_CONFIG
from ronin_pedroam_6d import prepare
from tracker.scekf import ImuMSCKF


class TlioAdapterTests(unittest.TestCase):
    def test_same_imu_grid_and_initial_attitude_as_ronin_without_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            n=1401
            ns=1_770_000_000_000_000_000+np.arange(n,dtype=np.int64)*10_000_000
            gyro=np.tile([0.,0.,.08],(n,1))
            accel=np.tile([2.,3.,np.sqrt(9.81**2-13)],(n,1))
            for name,values in [('Gyroscope.csv',gyro),('TotalAcceleration.csv',accel)]:
                pd.DataFrame(dict(time=ns,x=values[:,0],y=values[:,1],z=values[:,2])).to_csv(folder/name,index=False)
            (folder/'GyroscopeUncalibrated.csv').write_text('invalid; must not be read')
            (folder/'Orientation.csv').write_text('invalid; must not be read')
            times,g,a,r,indices,quality=prepare_body(folder)
            ronin_ns,imu,starts,_,diag=prepare(folder)
            np.testing.assert_array_equal(times,ronin_ns)
            np.testing.assert_array_equal(times[indices],ronin_ns[starts]+500_000_000)
            expected=Rotation.from_quat(diag['quat_wxyz'][1000,[1,2,3,0]]).as_matrix()
            np.testing.assert_allclose(r,expected,atol=1e-13)
            np.testing.assert_allclose(r@g[2000],imu[2000,:3],atol=1e-7)
            np.testing.assert_allclose(r@a[2000],imu[2000,3:],atol=1e-6)
            # Changing future IMU samples cannot change the initialized attitude.
            changed=pd.read_csv(folder/'TotalAcceleration.csv')
            changed.loc[1100:,'x']+=100.
            changed.to_csv(folder/'TotalAcceleration.csv',index=False)
            np.testing.assert_array_equal(prepare_body(folder)[3],r)

    def test_stationary_gravity_and_bias_sign_in_ekf_propagation(self):
        r=Rotation.from_euler('xyz',[.2,-.3,.7]).as_matrix()
        bias_g=np.array([[.01],[-.02],[.005]])
        bias_a=np.array([[.1],[-.05],[.03]])
        acc=r.T@np.array([[0.],[0.],[9.81]])+bias_a
        f=ImuMSCKF(SimpleNamespace(**FILTER_CONFIG))
        f.initialize_with_state(0,r,np.zeros((3,1)),np.zeros((3,1)),bias_a,bias_g)
        for i in range(1,201):
            f.propagate(acc,bias_g,i*5000)
        np.testing.assert_allclose(f.state.s_p,0,atol=1e-12)
        np.testing.assert_allclose(f.state.s_v,0,atol=1e-12)
        np.testing.assert_allclose(f.state.s_R,r,atol=1e-12)


if __name__=='__main__':
    unittest.main()
