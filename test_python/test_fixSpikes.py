from helper import *
import pytest

def test_fixSpikes():
    R1 = 2
    R2_1 = 1
    R2_2 = 2.5
    torus = mrmesh.makeTorusWithSpikes(R1, R2_1, R2_2, 10, 10, None)

    # (5 =~ 3pi/2) rad - minSumAngle
    mrmesh.removeSpikes(torus, 3, 5, None)

    # now all points are in that range from the center
    # comment 'remove spikes' to catch this assert
    for i in torus.points.vec:
        assert (i.x*i.x + i.y*i.y + i.z*i.z < (R1*R1 + R2_1*R2_1) * 2.5 )
