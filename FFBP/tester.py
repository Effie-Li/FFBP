import glob, os
import numpy as np

from tensorflow.tensorboard.backend.event_processing.event_accumulator import EventAccumulator

event_file = '/Users/alexten/Projects/pdpyflow/xor/train/log3/'
histo_str = 'weights'

eacc = EventAccumulator(event_file)
eacc.Reload()
print(eacc.Tags())

lc = np.stack(
  [np.asarray([scalar.step, scalar.value])
  for scalar in ea.Histograms(histo_str)])

