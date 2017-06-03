import numpy as np
from scipy.signal import butter, lfilter, freqz
from matplotlib import pyplot as plt

class ButterworthFilter:

	def __init__(self, sample_input, fc = 0.1, order = 2):
		self.signal_dim = sample_input.shape
		# self.signal_ndim = self.signal_dim.size
		# print "signal dimensions : ",  self.signal_dim
		t1 = [order+1] + [1 for x in self.signal_dim]
		t2 = [order] + [1 for x in self.signal_dim]
		self.past_input = np.tile(sample_input,tuple(t1))
		self.past_output = np.tile(sample_input,tuple(t2))
		# print self.past_output
		# print self.past_input
		self.order = order
		self.fc = fc
		self.computeCoeff(fc, order)
		# print self.output_coeffs
		# print self.input_coeffs

	def computeCoeff(self, fc, order):
		b, a = butter(order, fc*0.5, btype='low', analog=False)
		# print "output coeffs, input coeffs : ", a, b
		t1 = list(self.signal_dim) + [1]
		self.output_coeffs = np.tile(a[1::],tuple(t1)).T.reshape(self.past_output.shape)
		self.input_coeffs = np.tile(b,tuple(t1)).T.reshape(self.past_input.shape)

	def updateInput(self, newInput):
		self.past_input[1::] = self.past_input[0:-1]
		self.past_input[0] = newInput

	def updateOutput(self, newOutput):
		self.past_output[1::] = self.past_output[0:-1]
		self.past_output[0] = newOutput

	def updateFilter(self, newInput):
		self.updateInput(newInput)
		newOutput = np.sum(self.input_coeffs*self.past_input,0) - np.sum(self.output_coeffs*self.past_output,0)
		self.updateOutput(newOutput)
		return newOutput

class ProximityFilter:

	def __init__(self, first_input, treshold):
		self.treshold = treshold
		self.previous_value = first_input

	def updateFilter(self, newInput):
		if(np.linalg.norm(newInput - self.previous_value) > self.treshold):
			return self.previous_value
		else:
			self.previous_value = newInput
			return newInput

if __name__ == '__main__':

	signal = np.ones((250,2))
	signal[12,0] = 10
	signal[45,1] = 10
	signal[15,1] = 0.5
	signal[46,0] = 0.2
	filtered_signal = np.zeros((250,2))

	fb = ButterworthFilter(signal[0], 0.5, 2)
	fp = ProximityFilter(signal[0], 1)

	for i in range(250):
		tmp = fp.updateFilter(signal[i])
		filtered_signal[i] = fb.updateFilter(tmp)

	plt.figure(0)
	plt.plot(signal)
	plt.axis([0,250,0,12])

	plt.figure(1)
	plt.plot(filtered_signal)
	plt.axis([0,250,0,12])
	plt.show()
