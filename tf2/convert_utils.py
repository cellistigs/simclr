## A note on this module. This module provides functions that help us to translate between simclr style weights, and weights that we can use in our pose tracking algorithms downstream. SimCLR has a tf1 and tf2 implementation (both of which correspond to SimCLRv2), but both of them accept as input tf1 style weights. However, they output tf2 style weights. The conversions here are intended to be used for tf1 style "saver" checkpoints. Key note however, is that *even though tf2 style simclr uses tf1 style checkpoint, the resnet weights nave different names.* Therefore, when a variable below says "to_simclr_tf2", this refers to the tf2 style resnet weight naming, not the actual checkpoint format.  

import os 
import math


def var_slim_to_simclr(var):
    """Thin wrapper for slim_to_simclr_v1 below- takes the tf variable as input instead of the variable name. 
    UPDATE: this also works for simclr v2. 

    :param var: tensorflow variable. 
    """
    return slim_to_simclr_tf1(var.op.name)

def var_slim_to_simclr_tf2(var):
    """Thin wrapper for slim_to_simclr_tf2 below- takes the tf variable as input instead of the variable name. 

    :param var: tensorflow variable. 
    """
    return slim_to_simclr_tf2(var.op.name)

def unit_mapping(unit,kern):
    """Given a unit and kernel, gives flat index into convolution. (starts at 1)

    :param unit: unit integer (1-{3,4,6}, depending on block.)
    :param kern: kernel string: "shortcut,conv1,conv2,conv3"
    :returns: flat index into unit within the block.  
    """
    if unit == 1: 
        mapping = {"shortcut":0,
                   "conv1":1,
                   "conv2":2,
                   "conv3":3}
    else:    
        offset = 4+3*(unit-2) # offset by 4 for the first unit, plus 3 for every additional unit. 
        mapping = {"conv1":0+offset,
                   "conv2":1+offset,
                   "conv3":2+offset}
    return mapping[kern]+1    

def inv_unit_mapping_tf2(block_index,rel_index):
    """Given the block index and the relative index of the layer, returns the global unit index (0-15) as required by tf2: i.e. this layer is layer {rel_index} in block {block_index}

    :param block_index: 1-4
    :param rel_index: 1-19, depending on the block 
    """


def slim_to_simclr_tf1(var_name):
    """ Given the slim style parameter names, return the simclr-v1/v2 resnet 50 style parameter names for tensorflow 1.  

    Assumes that the slim model has the following variable organization: 
    - resnet_v1_50
      - mean_rgb
      - logits (we don't care)
      - conv1
        - BatchNorm
          - beta
          - gamma
          - moving_mean
          - moving_variance
        - weights
      - block(1-4)
        - (block1:unit_(1-3),block2:unit_(1-4),block3:unit_(1-6),block4:unit_(1-3))
          - bottleneck_v1
              - (shortcut,conv1,conv2,conv3)
                - BatchNorm
                  - beta
                  - gamma
                  - moving_mean
                  - moving_variance
                - weights  
   
    Furthermore, assumes that the ssl model has the following organization: 
    - base_model
      - conv2d(,_1..._52)
        - kernel
      - batch_normalization(,_1...,_52)
        - beta
        - gamma
        - moving_mean
        - moving_variance
    - head_supervised
      - (we don't care)

    :param var_name: the variable name in slim style formatting. See test_compare_vars_shapes_simclr for documentation of formatting. Assume these are resnet_v1_50 prefixed params. 
    """
    block_mapping = {1:0,
                     2:3*3+1,
                     3:7*3+2,
                     4:13*3+3} ## how many layers are included before this block? equal to number of units*3 + number of blocks (we add an additional shortcut "layer" for each block.)
    # handle corner cases 
    if "resnet_v1_50/conv1" in var_name: 
        resnet,conv1,info = var_name.split("/",2)
        ssl_nb = "" ## for formatting: initial convs take no underscore index. 
    else:   
        ## split the path: 
        resnet,block,unit,bottleneck,kernel,info = var_name.split("/",5)
        block_nb = int(block.split("block")[-1])
        unit_nb = int(unit.split("unit_")[-1])

        ssl_nb = "_" + str(block_mapping[block_nb]+unit_mapping(unit_nb,kernel))
    
    if "BatchNorm" in info:
        param = info.split("/")[-1]
        string = "base_model/batch_normalization{}/"+param
    elif "weights" in info:    
        string = "base_model/conv2d{}/kernel"
    return string.format(ssl_nb)    

def slim_to_simclr_tf2(var_name):
    """ Given the slim style parameter names, return the simclr-v1/v2 resnet 50 style parameter names for tensorflow 2.  

    Assumes that the slim model has the following variable organization: 
    - resnet_v1_50
      - mean_rgb
      - logits (we don't care)
      - conv1
        - BatchNorm
          - beta
          - gamma
          - moving_mean
          - moving_variance
        - weights
      - block(1-4)
        - (block1:unit_(1-3),block2:unit_(1-4),block3:unit_(1-6),block4:unit_(1-3))
          - bottleneck_v1
              - (shortcut,conv1,conv2,conv3)
                - BatchNorm
                  - beta
                  - gamma
                  - moving_mean
                  - moving_variance
                - weights  
   
    Furthermore, assumes that the ssl model has the following organization: 
    - resnet
      - conv2d_fixed_padding
        - conv2d
          - kernel:0
      - batch_norm_relu    
        - sync_batch_normalization
          - gamma:0
          - beta:0
      - block_group{1,2,3,4}
        - bottleneck_block{"",_{1,2,...15}}
          - conv_2d_fixed_padding_{1,2,...52}
            - conv_2d_{1,2,...52}
              - kernel:0
          - batch_norm_relu_{1,2,...52}
            - sync_batch_normalization_{1,2,...52}
              - gamma:0
              - beta:0
      Here, the bottleneck block indices are continuous over the variable block numbers.         

    :param var_name: the variable name in slim style formatting. See test_compare_vars_shapes_simclr for documentation of formatting. Assume these are resnet_v1_50 prefixed params. 
    """
    bottleneck_block_mapping = {1:0,
                                2:3,
                                3:7,
                                4:13}
    block_mapping = {1:0,
                     2:3*3+1,
                     3:7*3+2,
                     4:13*3+3} ## how many layers are included before this block? equal to number of units*3 + number of blocks (we add an additional shortcut "layer" for each block.)
    # handle corner cases 
    if "resnet_v1_50/conv1" in var_name: 
        resnet,conv1,info = var_name.split("/",2)
        ssl_nb = "" ## for formatting: initial convs take no underscore index. 
        
        
    else:   
        ## split the path: 
        resnet,block,unit,bottleneck,kernel,info = var_name.split("/",5)
        block_nb = int(block.split("block")[-1])
        unit_nb = int(unit.split("unit_")[-1])
        bottleneck_ind = unit_nb+bottleneck_block_mapping[block_nb] #1-16

        ssl_nb = "_" + str(block_mapping[block_nb]+unit_mapping(unit_nb,kernel))
        if bottleneck_ind == 1:
            bottleneck_nb = ""
        else:    
            bottleneck_nb = "_" + str(bottleneck_ind-1)
    
    path_components = ["resnet"]
    if ssl_nb == "":
        pass
    else: 
        path_components.append("block_group{block}")
        path_components.append("bottleneck_block{bottle}")

    if "BatchNorm" in info:
        param = info.split("/")[-1]
        path_components.append("batch_norm_relu{ssl}")
        path_components.append("sync_batch_normalization{ssl}")
        path_components.append(param+"")
    elif "weights" in info:    
        path_components.append("conv2d_fixed_padding{ssl}")
        path_components.append("conv2d{ssl}")
        path_components.append("kernel")
    if ssl_nb == "":
        string = os.path.join(*path_components).format(ssl = ssl_nb)
    else:    
        string = os.path.join(*path_components).format(block=block_nb,bottle=bottleneck_nb,ssl = ssl_nb)
    return string

def simclr_tf1_to_simclr_tf2(var_name):
    """ Given the simclr tf1 style parameter names, return the simclr tf2 style parameter names. 
    Assumes the tf1 style parameter names are as follows: 
    - base_model
      - conv2d(,_1..._52)
        - kernel
      - batch_normalization(,_1...,_52)
        - beta
        - gamma
        - moving_mean
        - moving_variance
    - head_supervised
      - (we don't care)

    Furthermore, assumes that the tf2 style model has the following organization: 
    - resnet
      - conv2d_fixed_padding
        - conv2d
          - kernel:0
      - batch_norm_relu    
        - sync_batch_normalization
          - gamma:0
          - beta:0
      - block_group{1,2,3,4}
        - bottleneck_block{,_{1,2,...15}}
          - conv_2d_fixed_padding_{1,2,...52}
            - conv_2d_{1,2,...52}
              - kernel:0
          - batch_norm_relu_{1,2,...52}
            - sync_batch_normalization_{1,2,...52}
              - gamma:0
              - beta:0
    :param var_name: the variable name in tf1 style format. 
    """
    ## parse out ssl_nb:
    resnet,op,info = var_name.split("/",2)
    op_spec = op.split("_")

    ## handle corner case: 
    try: 
        ssl_index = int(op_spec[-1])
        ssl_nb = "_{}".format(ssl_index)
        index_params = flat_index_to_hierarchical_tf2(ssl_index)
        block_nb = index_params["block_index"]
        if index_params["bottleneck_index"] == 0:
            bottleneck_nb = ""
        else:    
            bottleneck_nb = "_{}".format(index_params["bottleneck_index"])
    except ValueError:         
        resnet,conv1,info = var_name.split("/",2)
        ssl_nb = "" ## for formatting: initial convs take no underscore index. 
            
    path_components = ["resnet"]
    if ssl_nb == "":
        pass
    else: 
        path_components.append("block_group{block}")
        path_components.append("bottleneck_block{bottle}")

    if "batch_normalization" in op:
        param = info.split("/")[-1]
        path_components.append("batch_norm_relu{ssl}")
        path_components.append("sync_batch_normalization{ssl}")
        path_components.append(param+"")
    elif "conv2d" in op:    
        path_components.append("conv2d_fixed_padding{ssl}")
        path_components.append("conv2d{ssl}")
        path_components.append("kernel")
    if ssl_nb == "":
        string = os.path.join(*path_components).format(ssl = ssl_nb)
    else:    
        string = os.path.join(*path_components).format(block=block_nb,bottle=bottleneck_nb,ssl = ssl_nb)
    return string    


def flat_index_to_hierarchical_tf2(ssl_nb): 
    """Given a flat index (1-52) of conv-batchnorm-relu components, returns the block (1-4) and bottlenemck (0-15) indices. 

    To recover the block mappings, we need to bound the ssl number by the threshold indices we know correspond to the blocks. I.e., if ssl_nb > 13*3+3, block_mapping = 4 

    :param ssl_nb: the flat index into conv-batchnorm-relu components. 
    :returns: {"block_index":block_index,"bottleneck_index":bottleneck_index} 
    """

    ## get the block mapping: 
    block_index,remain = get_inv_block(ssl_nb)
    bottleneck_index = get_inv_unit(block_index,remain)
    return {"block_index":block_index,"bottleneck_index":bottleneck_index}
    

def get_inv_block(ssl_nb):
    """
    :param ssl_nb: ind 1-52 of layers in the actual resnet blocks. 
    """
    assert ssl_nb >= 1; "must be 1 indexed."
    inv_block_mapping = {
            13*3+3:4,
            7*3+2:3,
            3*3+1:2,
            0:1,
            } ## how many layers are included before this block? equal to number of units*3 + number of blocks (we add an additional shortcut "layer" for each block.)

    for thresh,key in inv_block_mapping.items():
        diff = ssl_nb-1 - thresh
        if diff >=0:
            block_index = key
            break
        else:
            pass
    return block_index,diff    

def get_inv_unit(block_index,diff):
    """
    given a block index and a 0-indexed layer in that block, returns a unit index. 
    """
    bottleneck_block_mapping = {1:0,
                                2:3,
                                3:7,
                                4:13}
    return bottleneck_block_mapping[block_index] + math.floor((abs(diff-1)/3))
