import sys
from contextlib import contextmanager
import os
import numpy as np

from alfred import init_class

testdir = os.path.dirname(os.path.realpath(__file__))+'/'


@contextmanager
def replace_stdin(target):
    orig = sys.stdin
    sys.stdin = target
    yield
    sys.stdin = orig


def test_init_lcs():
    """Tests initializing an Init_lcs object, building it with user inputs, saving the file, and loading it back in.
    """

    with replace_stdin(open(testdir+'init_class_in/init_lcs_in.txt')):

        try:

            init_class.Init_lcs(testdir).create()

            init_lcs = init_class.Init_lcs(testdir).from_file()

            assert len(init_lcs.table) == 1
            assert (np.array(list(init_lcs.table[0])) == np.array(['fake_lc.csv','fake data','time','flux','err','q',0,1.0,120.0,'filt',False])).all()

        finally:
            if os.path.exists(testdir+'init_lcs.txt'):
                os.remove(testdir+'init_lcs.txt')


def test_init_rv():
    """Tests initializing an Init_rv object, building it with user inputs, saving the file, and loading it back in.
    """

    with replace_stdin(open(testdir+'init_class_in/init_rv_in.txt')):

        try:

            init_class.Init_rv(testdir).create()

            init_rv = init_class.Init_rv(testdir).from_file()

            assert len(init_rv.table) == 1
            assert (np.array(list(init_rv.table[0])) == np.array(['fake_rv.csv','fake','time','rv','err',0,1.0,'m/s'])).all()

        finally:
            if os.path.exists(testdir+'init_rv.txt'):
                os.remove(testdir+'init_rv.txt')


def test_init_star():
    """Tests initializing an Init_star object, building it with user inputs, saving the file, and loading it back in.
    """

    with replace_stdin(open(testdir+'init_class_in/init_star_in.txt')):

        try:

            init_class.Init_star(testdir).create()

            init_star = init_class.Init_star(testdir).from_file()

            assert len(init_star.table) == 2
            assert (np.array(list(init_star.table[0])) == np.array(['Value',1.0,1.0,5800,4.5,0.0,100.0,8.0,8.0,8.0,8.0,8.0,8.0,8.0,8.0,8.0])).all()
            assert (np.array(list(init_star.table[1])) == np.array(['Error',0.1,0.1,50,0.5,0.1,1.0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])).all()

        finally:
            if os.path.exists(testdir+'init_star.txt'):
                os.remove(testdir+'init_star.txt')


def test_init_ld():
    """Tests initializing an Init_ld object, building it with user inputs, saving the file, and loading it back in.
    """

    with replace_stdin(open(testdir+'init_class_in/init_ld_in.txt')):

        try:

            init_class.Init_ld(testdir).create()

            init_ld = init_class.Init_ld(testdir).from_file()

            assert len(init_ld.table) == 1
            assert (np.array(list(init_ld.table[0])) == np.array([0.5,0.5,'TESS'])).all()
        
        finally:
            if os.path.exists(testdir+'init_ld.txt'):
                os.remove(testdir+'init_ld.txt')


def test_init_planets():
    """Tests initializing an Init_planets object, building it with user inputs, saving the file, and loading it back in.
    """

    with replace_stdin(open(testdir+'init_class_in/init_planets_in.txt')):

        try:

            init_class.Init_planets(testdir).create()

            init_planets = init_class.Init_planets(testdir).from_file()

            assert len(init_planets.table) == 1
            assert (np.array(list(init_planets.table[0])) == np.array([True,True,False,True,5.5,1000.0,0.1,10,0.01,10,0.1,0.1])).all()

        finally:
            if os.path.exists(testdir+'init_planets.txt'):
                os.remove(testdir+'init_planets.txt')


def test_init_ttvs():
    """Tests initializing an Init_ttvs object, building it with user inputs, saving the file, and loading it back in.
    """

    with replace_stdin(open(testdir+'init_class_in/init_ttvs_in.txt')):

        try:

            init_class.Init_ttvs(testdir).create()

            init_ttvs = init_class.Init_ttvs(testdir).from_file()

            assert len(init_ttvs.table) == 3
            assert (np.array(list(init_ttvs.table[0])) == np.array([1.0,1.0,1.0])).all()
            assert (np.array(list(init_ttvs.table[1])) == np.array([2.0,2.0,2.0])).all()
            assert init_ttvs.table['2'][2] == 3.0
            assert np.isnan(list(init_ttvs.table['1','3'][2])).all()

        finally:
            if os.path.exists(testdir+'init_ttvs.txt'):
                os.remove(testdir+'init_ttvs.txt')


def test_init_priors():
    """Tests initializing an Init_priors object, building it with user inputs, saving the file, and loading it back in.
    """

    with replace_stdin(open(testdir+'init_class_in/init_priors_in.txt')):
            
        try:

            init_class.Init_priors(testdir).create()

            init_priors = init_class.Init_priors(testdir).from_file()

            assert len(init_priors.table) == 2
            assert (np.array(list(init_priors.table[0])) == np.array(['log(P) 1','F',1.5,0.0])).all()
            assert (np.array(list(init_priors.table[1])) == np.array(['F0 x','G',1.0,0.01])).all()

        finally:
            if os.path.exists(testdir+'init_priors.txt'):
                os.remove(testdir+'init_priors.txt')