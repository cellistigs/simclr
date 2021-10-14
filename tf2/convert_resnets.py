## Based off of code in https://www.tensorflow.org/guide/migrate/migrating_checkpoints#checkpoint_conversion: convert a given resnet 50 checkpoint changing variable names as necessary.  
import pprint
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import click
import convert_utils 

def get_var_names(save_path):
  reader = tf.train.load_checkpoint(save_path)
  shapes = reader.get_variable_to_shape_map()
  dtypes = reader.get_variable_to_dtype_map()
  return list(shapes.keys())

def get_checkpoint(save_path):  
  reader = tf.train.load_checkpoint(save_path)
  shapes = reader.get_variable_to_shape_map()
  dtypes = reader.get_variable_to_dtype_map()
  print(f"Checkpoint at '{save_path}':")
  vardict = {}
  for key in shapes:
      vardict[key] = reader.get_tensor(key)
  return vardict    

def print_checkpoint(save_path):
  reader = tf.train.load_checkpoint(save_path)
  shapes = reader.get_variable_to_shape_map()
  dtypes = reader.get_variable_to_dtype_map()
  print(f"Checkpoint at '{save_path}':")
  for key in shapes:
    print(f"  (key='{key}', shape={shapes[key]}, dtype={dtypes[key].name}, "
          f"value={reader.get_tensor(key)})")

def convert_simclr_tf1(checkpoint_path,output_prefix,convert_func):
    """Converts a simclr tf1 _style TF1 checkpoint by changing names based on the given key mapping. 
    You've got to be careful because these weights have a "Momentum" parameter that can get assigned to kernel values from the EMA model of v1 simclr weights. 

    :param checkpoint_path: path to the tensorflow 1 checkpoint. 
    :param output_prefix: path prefix of the converted checkpoint.
    :returns: path to converted checkpoint. 
    """
    vars = {}
    reader = tf.train.load_checkpoint(checkpoint_path)
    dtypes = reader.get_variable_to_dtype_map()
    for key in dtypes.keys():
        if (key.startswith("base_model/conv2d") or key.startswith("base_model/batch_normalization")) and not key.endswith("Momentum"):
            vars[convert_func(key)] = tf.Variable(reader.get_tensor(key))
        else:    
            vars[key] = tf.Variable(reader.get_tensor(key))

    return tf1.train.Saver(var_list=vars).save(sess= None,save_path = output_prefix)    

def convert_slim(checkpoint_path,output_prefix,convert_func):
    """Converts a slim_style TF1 checkpoint by changing names based on the given key mapping. 

    :param checkpoint_path: path to the tensorflow 1 checkpoint. 
    :param output_prefix: path prefix of the converted checkpoint.
    :returns: path to converted checkpoint. 
    """
    vars = {}
    reader = tf.train.load_checkpoint(checkpoint_path)
    dtypes = reader.get_variable_to_dtype_map()
    for key in dtypes.keys():
        if "resnet_v1_50/conv" in key or "resnet_v1_50/block" in key:
            vars[convert_func(key)] = tf.Variable(reader.get_tensor(key))
        else:    
            vars[key] = tf.Variable(reader.get_tensor(key))

    return tf1.train.Saver(var_list=vars).save(sess= None,save_path = output_prefix)    

mappings = {
        "slim_to_simclr_tf2":{"names":convert_utils.slim_to_simclr_tf2,"converter":convert_slim},
        "slim_to_simclr_tf1":{"names":convert_utils.slim_to_simclr_tf1,"converter":convert_slim},
        "simclr_tf1_to_simclr_tf2":{"names":convert_utils.simclr_tf1_to_simclr_tf2,"converter":convert_simclr_tf1}
        }


@click.command("convert a tensorflow checkpoint from slim format to simclr_tf2 format. ")
@click.option("--checkpoint-path")
@click.option("--prefix")
@click.option("--mapping",help = "which resnet mapping to use. Options are [`slim_to_simclr_tf2`,`slim_to_simclr_tf1`, or `simclr_tf1_to_simclr_tf2`]")
def main(checkpoint_path,prefix,mapping):    

    old_vars = get_var_names(checkpoint_path)
    new_checkpoint = mappings[mapping]["converter"](checkpoint_path,prefix,mappings[mapping]["names"])
    new_vars = get_var_names(new_checkpoint)
    print("Old Vars:")
    pprint.pprint(old_vars)
    print("New Vars:")
    pprint.pprint(new_vars)

if __name__ == "__main__":
    main()


