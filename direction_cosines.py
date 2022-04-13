# TO-DO: Containerize to microservice
import numpy as np

# This microservice obtains the direction cosines for conversion between 
# ECEF, ECI, NED, and flight-body inertial coordinate systems
# Requires: Longitude, latitude, instance of metadata distributor
# Required by: Environment model, motion vectors relative to ground
# Outputs: Direction cosines matrices
class DirectionCosines():
  def __init__(self, metadata_distributor):
    self.metadata_distributor = metadata_distributor
    self.constants = self.metadata_distributor.constants
    self.q = self.metadata_distributor.get_var('q')
    self.lamb = self.metadata_distributor.get_var('lamb')
    self.eta = self.metadata_distributor.get_var('eta')
  
    # t_sb: transformation matrix of ECI to local coordinate of flight body
    def get_t_sb(self):
      q1, q2, q3, q4 = self.q
      t_sb = np.array([[q1**2-q2**2-q3**2+q4**2, 2*(q1*q2+q3*q4), 2*(q1*q3-q2*q4)],
                      [2*(q1*q2-q3*q4), -q1**2+q2**2-q3**2+q4**2, 2*(q2*q3+q1*q4)],
                      [2*(q1*q3+q2*q4), 2*(q2*q3-q1*q4), -q1**2-q2**2+q3**2+q4**2]])
      self.metadata_distributor.set({'t_sb': t_sb})
      return t_sb

    # t_sh: transformation of ECI to NED coordinates
    def get_t_sh(self, t):
      t_sh = np.array([[-np.sin(self.eta)*np.cos(self.constants.OMEGA*t+self.lamb), -np.sin(self.eta)*np.sin(self.constants.OMEGA*t+self.lamb), np.cos(self.eta)],
                      [-np.sin(self.constants.OMEGA*t+self.lamb), np.cos(self.constants.OMEGA*t+self.lamb), 0],
                      [-np.cos(self.eta*np.cos(self.constants.OMEGA*t+self.lamb)), -np.cos(self.eta)*np.sin(self.constants.OMEGA*t+self.lamb), -np.sin(self.eta)]])
      self.metadata_distributor.set({'t_sh': t_sh})
      return t_sh

    # t_hb: transformation of NED to flight body coorindates
    def get_t_hb(self):
      t_sb = self.metadata_distributor.get_var('t_sb')
      t_sh = self.metadata_distributor.get_var('t_sh')
      t_hb = np.matmul(t_sb, np.linalg.inv(t_sh))
      self.metadata_distributor.set({'t_sb': t_hb})
      return t_hb