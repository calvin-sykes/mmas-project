import glob
import numpy as np
from astropy.table import Table

class CloudyPlotter:
    def __init__(self, filename_base):
        # Get a sorted list of all the files produced by the run 
        file_list = sorted(glob.glob('grid*_' + filename_base + '.ovr'))
        #ems_list = sorted(glob.glob('grid*_' + filename_base + '.ems'))
        # Get the grid file, which describes the parameters used in each file
        self.grid_file = np.loadtxt(filename_base + '.grd',
                                    usecols=(0,6),
                                    dtype=np.dtype([('idx' ,np.int),
                                                    ('hden',np.float)]))
        self.data = []
        header = []
        for idx, file in enumerate(file_list):
            # only the first file has a header
            if idx == 0:
                header = Table.read(file, format='ascii').colnames
            self.data.append(Table.read(file, format='ascii', names=header))
                #np.genfromtxt(file, usecols=range(0,8)))

    def get_col(self, filenum, colname):
        assert filenum in range(0, len(self.data))
        return self.data[filenum][colname]

    def get_grid_param(self, filenum, paramname):
        assert filenum in range(0, len(self.grid_file))
        return self.grid_file[paramname][filenum]

    def nfiles(self):
        return len(self.data)
