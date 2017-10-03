from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np


def get_event_accumulator(path_to_event_file):
    event_file = path_to_event_file
    eacc = EventAccumulator(event_file)
    eacc.Reload()
    return eacc


def retrieve_array(path_to_event_file, layer_name, tensor_name, data_type=np.float32):
    eacc = get_event_accumulator(path_to_event_file)
    dt = data_type
    for tt in eacc.Tags()['tensors']:
        fetch = '/'.join([layer_name, tensor_name])
        if fetch in tt:
            print(tt) #####
            ra = []
            rs = []
            
            for te in eacc.Tensors(tt):
                shape = [i.size for i in te.tensor_proto.tensor_shape.dim]
                print(shape) #####
                binary_tensor = te.tensor_proto.tensor_content
                deci_tensor  = np.fromstring(binary_tensor, dtype=dt).reshape(shape)
                ra.append(deci_tensor)
                rs.append(te.step)
    return (ra, rs)


def retrieve_array2(path_to_event_file, layer_name, tensor_name, data_type=np.float32):
    eacc = get_event_accumulator(path_to_event_file)
    dt = data_type
    params_fetch = '/'.join([layer_name, tensor_name, 'params_summary'])
    data_fetch = '/'.join([layer_name, tensor_name, 'data_summary'])
    for tt in eacc.Tags()['tensors']:
        if params_fetch in tt:
            snap = {}
            for te in eacc.Tensors(tt):
                shape = [i.size for i in te.tensor_proto.tensor_shape.dim]
                binary_tensor = te.tensor_proto.tensor_content
                deci_tensor  = np.fromstring(binary_tensor, dtype=dt).reshape(shape)
                snap[te.step] = deci_tensor
        elif data_fetch in tt:
            snap = {}
            for te in eacc.Tensors(tt):
                shape = [i.size for i in te.tensor_proto.tensor_shape.dim]
                binary_tensor = te.tensor_proto.tensor_content
                deci_tensor = np.fromstring(binary_tensor, dtype=dt).reshape(shape)
                snap[te.step] = deci_tensor
    return snap


def retrieve_loss(path_to_event_file):
    eacc = get_event_accumulator(path_to_event_file)
    loss_log = np.stack([np.asarray([scalar.step, scalar.value]) for scalar in eacc.Scalars('train/error_summary')])
    return loss_log
