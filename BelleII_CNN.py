import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import tensorflow.compat.v2 as tf
import math

"""
STATUS:

The fundction to generate the correct window_position matrix is tested 
it is ok to use!!!!!!

"""
## This Class defines the CDC wire configuration
class BelleCdcWire:
    layers = 8 * [160] + \
             6 * [160] + \
             6 * [192] + \
             6 * [224] + \
             6 * [256] + \
             6 * [288] + \
             6 * [320] + \
             6 * [352] + \
             6 * [384]

    # layerBaseId = itertools.accumulate(layers)  #I dont know why , but this may has version issue, it doesnt return the correct array
    layerBaseId = [160, 320, 480, 640, 800,
                   960, 1120, 1280, 1440, 1600, 1760, 1920,
                   2080, 2240, 2432, 2624, 2816, 3008, 3200, 3392,
                   3616, 3840, 4064, 4288, 4512, 4736, 4992, 5248, 5504,
                   5760, 6016, 6272, 6560, 6848, 7136, 7424, 7712, 8000, 8320,
                   8640, 8960, 9280, 9600, 9920, 10272, 10624, 10976, 11328, 11680,
                   12032, 12416, 12800, 13184, 13568, 13952, 14336]

    def __init__(self, wire: int, layer: int):
        self.id = self._calculateUniqueId(wire=wire, layer=layer)
        self.wire = wire
        self.layer = layer

    @classmethod
    def fromUniqueId(cls, id: int):
        layer, wire = cls._calculateLayerWire(id=id)
        return cls(wire=wire, layer=layer)

    @classmethod
    def _calculateUniqueId(cls, layer: int, wire: int) -> int:
        """
        This function returns a unique wire id based on the given wireId and layerId.
        The configuration of the Belle II detector is based on Technical Report. Layers
        are numbered from the interaction point to the outer shell layers.
        """

        if (layer < 0 or layer >= len(cls.layers)):
            raise IndexError(f"Given layer id {layer:d} is out of range [0,{len(cls.layers):d})")

        if (wire < 0 or wire >= cls.layers[layer]):
            raise IndexError(f"Given wire id {wire:d} is out of range[0,{cls.layers[layer]:d})")

        return wire + sum(cls.layers[:layer])

    @classmethod
    def _calculateLayerWire(cls, id: int):
        if (id < 0 or id >= sum(cls.layers)):
            raise IndexError(f"Given unique id {id:d} is out of range [0,{sum(cls.layers):d})")

        for i, baseId in enumerate(cls.layerBaseId):
            # for i,baseId in enumerate(cls.layers):
            if (baseId > id):
                layerId = i
                break

        wireId = id - sum(cls.layers[:layerId])

        return (layerId, wireId)
    @classmethod
    def _calculateSuperlayer_from_id(cls,id:int):
        iclayer , _ = cls._calculateLayerWire(id)
        if iclayer <= 7:
            return int(0)
        elif iclayer <= 13:
            return 1
        elif iclayer <= 19:
            return 2
        elif iclayer <= 25:
            return 3
        elif iclayer <= 31:
            return 4
        elif iclayer <= 37:
            return 5
        elif iclayer <= 43:
            return 6
        elif iclayer <= 49:
            return 7
        else:
            return 8
    @classmethod
    def _calculateSuperlayer_from_layer(cls,iclayer:int):
        if iclayer <= 7:
            return int(0)
        elif iclayer <= 13:
            return 1
        elif iclayer <= 19:
            return 2
        elif iclayer <= 25:
            return 3
        elif iclayer <= 31:
            return 4
        elif iclayer <= 37:
            return 5
        elif iclayer <= 43:
            return 6
        elif iclayer <= 49:
            return 7
        else:
            return 8
    @classmethod
    def _return_wireid_xy_lut(cls):
        df = pd.read_csv("cdcwire_config.txt")
        df = df[["x","y"]]
        temp = df.to_numpy()
        return  temp

################ Plot Methods ##################
def _plot_cross_section(
        plot_layers={   
                0:[0,1,2,3,4,5,6,7],
                1:[0,1,2,3,4,5], 
                2:[0,1,2,3,4,5], 
                3:[0,1,2,3,4,5], 
                4:[0,1,2,3,4,5], 
                5:[0,1,2,3,4,5], 
                6:[0,1,2,3,4,5], 
                7:[0,1,2,3,4,5], 
                8:[0,1,2,3,4,5]}
):
    Super_Layer_marker = {0: "+", 1: "o", 2: "+", 3: "o", 4: "+", 5: "o", 6: "+", 7: "o", 8: "+"}
    colors = ['#666A98', '#866AA0', '#A8679E', '#E781A1', '#FF9B87', '#C0BC84', '#9BDE7E', '#039590', '#2F4858']
    df = pd.read_csv("cdcwire_config.txt")
    df = df[["superlayer","ilayer" ,"x", "y"]]
    for superlayer in plot_layers:
        temp = df[(df["superlayer"] == superlayer)]
        for Ilayer in plot_layers[superlayer]:
            temp_Ilayer = temp[temp.ilayer == Ilayer]
            plt.plot(temp_Ilayer.x, temp_Ilayer.y, Super_Layer_marker[superlayer], markersize=1,color = colors[superlayer] )
def _plot_cross_section_x1y1():
    plot_layers={   
                0:[0,1,2,3,4,5,6,7],
                1:[0,1,2,3,4,5], 
                2:[0,1,2,3,4,5], 
                3:[0,1,2,3,4,5], 
                4:[0,1,2,3,4,5], 
                5:[0,1,2,3,4,5], 
                6:[0,1,2,3,4,5], 
                7:[0,1,2,3,4,5], 
                8:[0,1,2,3,4,5]}
    Super_Layer_marker = {0: "x", 1: "o", 2: "x", 3: "o", 4: "x", 5: "o", 6: "x", 7: "o", 8: "x"}
    colors = ['#666A98', '#866AA0', '#A8679E', '#E781A1', '#FF9B87', '#C0BC84', '#9BDE7E', '#039590', '#2F4858']
    df = pd.read_csv("cdcwire_config.txt")
    df = df[["superlayer","ilayer" ,"x", "y"]]
    for superlayer in plot_layers:
        temp = df[(df["superlayer"] == superlayer)]
        for Ilayer in plot_layers[superlayer]:
            temp_Ilayer = temp[temp.ilayer == Ilayer]
            print(temp_Ilayer.x.to_numpy())
            print(temp_Ilayer.y.to_numpy())
            temp_position_x = []
            temp_position_y = [] 
            for idx, temp_x in enumerate(temp_Ilayer.x.to_numpy()):
                if temp_x >= 0:
                    if temp_Ilayer.y.to_numpy()[idx] >= 0:
                        temp_position_x.append(temp_x)
                        temp_position_y.append(temp_Ilayer.y.to_numpy()[idx]) 
            
            plt.plot(temp_position_x, temp_position_y, Super_Layer_marker[superlayer], markersize=4,color = colors[superlayer] )
    layers_config = [160,160,192,224,256,288,320,352,384] 
    superLayer_size = [8,6,6,6,6,6,6,6,6] 
    superlayer_type = ["Axial","stereo","Axial","stereo","Axial","stereo","Axial","stereo","Axial"] 
    for i in range(9):
        plt.text(15 + i*11 ,-1.5,f"Superlayer {i}",fontsize=11)
        plt.text(15 + i*11 ,-5,f"{superlayer_type[i]} \n{superLayer_size[i]} * {layers_config[i] } ",fontsize=10)
        #plt.text(17 + i*11, -5, f"{superLayer_size[i]} * {layers_config[i]}",fontsize=12)
def _plot_cross_section_edge():
    _plot_cross_section({0:[0,7],
 1:[0,5],
 2:[0,5],
 3:[0,5],
 4:[0,5],
 5:[0,5],
 6:[0,5],
 7:[0,5],
 8:[0,5]
})


def _plot_mark_from_wireid(
        wireid_list = [],  # [wireid1,wireid2,wireid3,wireid4.......]must pass in a 1d arrary
        markersize = 4,
        color = "red",
        marker = "x"
):
    lut = BelleCdcWire._return_wireid_xy_lut()

    temp_xy = np.zeros((len(wireid_list), 2))
    for idx, wireid in enumerate(wireid_list):
        temp_xy[idx] = lut[wireid]
    plt.plot(temp_xy[:, 0], temp_xy[:, 1], marker, markersize=markersize, color=color)
    
def _plot_mark_from_pos(
        hits_list=[[]],   # defined as a 2d array [[x_pos,y_pos]......]
        color = "red",
        markersize = 4,
        marker = "x"
):
    plt.plot(hits_list[:,0],hits_list[:,1],marker,markersize=markersize, color=color)

class FPGA_hit_image_converter():
    def __init__(self):
        
        # defined as Lining_dict[wireID] = (index_from_offset, layer) 
        self.Lining_dict = self._create_Lining_dict()
        

    def _create_Lining_dict(self):
        
        Push_up_offset   = 3
        Pull_down_offset = 3
        Lining_dict = {}
        for wireid in range(14336):
    
            _layer,_wire = BelleCdcWire._calculateLayerWire(wireid)
            _index_from_offset =  _wire
    
            if BelleCdcWire._calculateSuperlayer_from_id(wireid) == 3 :       # Push Up On Superlayer 3       # Max wire 224    
                
                if _wire >= Push_up_offset:
                    _index_from_offset = _wire - Push_up_offset
                else:
                    _index_from_offset = 224 - (Push_up_offset - _wire)
            
            elif BelleCdcWire._calculateSuperlayer_from_id(wireid) == 7 :    # Push Up on Superlayer 7     # Max wire 352
            
                if _wire >= Push_up_offset:
                    _index_from_offset = _wire - Push_up_offset
                else:
                    _index_from_offset = 352 - (Push_up_offset - _wire)
                
            elif BelleCdcWire._calculateSuperlayer_from_id(wireid) == 1 :     # Pull down on Superlayer 1   # Max wire 160
                
                if _wire >=  160 - Pull_down_offset:
                    _index_from_offset = _wire - (160 - Pull_down_offset) 
                else:
                    _index_from_offset = _wire + Pull_down_offset
            
            elif BelleCdcWire._calculateSuperlayer_from_id(wireid) == 5 :    # Pull down on Superlayer 5    # Max wire 288
            
                if _wire >= 288 - Pull_down_offset:
                    _index_from_offset = _wire - (288 - Pull_down_offset)
                else:
                    _index_from_offset = _wire + Pull_down_offset
             
            Lining_dict[wireid] = (_index_from_offset,_layer)
        return Lining_dict
        
    def OR_Gate_mapping(self,
        index_from_offset, # Int
        layer              # Int
        ):
        """
        This Function implements the Squeeze methode, translate the CDC hits from circle to the squre
        """
        _super_layer = BelleCdcWire._calculateSuperlayer_from_layer(layer)
        y = layer
    
        if _super_layer == 0:                                                              ####### 0
        
            unsqueezed_bitwidth = 5 
            segment_ID = int(index_from_offset / unsqueezed_bitwidth)
            position_in_segment = index_from_offset % unsqueezed_bitwidth
        
            if position_in_segment == 0:
                embedding = 0
            elif position_in_segment == 1:
                embedding = 1
            elif position_in_segment == 2:
                embedding = 2
            elif position_in_segment == 3:
                embedding = 3
            elif position_in_segment == 4:
                embedding = 4

            x = segment_ID * 5 + embedding
        
        elif _super_layer == 1:                                                         #######  1
            unsqueezed_bitwidth = 5 
            segment_ID = int(index_from_offset / unsqueezed_bitwidth)
            position_in_segment = index_from_offset % unsqueezed_bitwidth
        
            if position_in_segment == 0:
                embedding = 0
            elif position_in_segment == 1:
                embedding = 1
            elif position_in_segment == 2:
                embedding = 2
            elif position_in_segment == 3:
                embedding = 3
            elif position_in_segment == 4:
                embedding = 4

            x = segment_ID * 5 + embedding
            # x =  index_from_offset
        elif _super_layer == 2:
        
            unsqueezed_bitwidth = 6 
            segment_ID = int(index_from_offset / unsqueezed_bitwidth)
            position_in_segment = index_from_offset % unsqueezed_bitwidth
        
            if position_in_segment == 0:
                embedding = 0
            elif position_in_segment == 1:
                embedding = 1
            elif position_in_segment == 2:
                embedding = 2
            elif position_in_segment == 3:
                embedding = 2
            elif position_in_segment == 4:
                embedding = 3
            elif position_in_segment == 5:
                embedding = 4
            x = segment_ID * 5 + embedding
        
        elif _super_layer == 3:
        
            unsqueezed_bitwidth = 7
            segment_ID = int(index_from_offset / unsqueezed_bitwidth)
            position_in_segment = index_from_offset % unsqueezed_bitwidth
        
            if position_in_segment == 0:
                embedding = 0
            elif position_in_segment == 1:
                embedding = 1
            elif position_in_segment == 2:
                embedding = 1
            elif position_in_segment == 3:
                embedding = 2
            elif position_in_segment == 4:
                embedding = 3
            elif position_in_segment == 5:
                embedding = 4
            elif position_in_segment == 6:
                embedding = 4
            x = segment_ID * 5 + embedding
        
        elif _super_layer == 4:
            unsqueezed_bitwidth = 8
            segment_ID = int(index_from_offset / unsqueezed_bitwidth)
            position_in_segment = index_from_offset % unsqueezed_bitwidth
        
            if position_in_segment == 0:
                embedding = 0
            elif position_in_segment == 1:
                embedding = 0
            elif position_in_segment == 2:
                embedding = 1
            elif position_in_segment == 3:
                embedding = 2
            elif position_in_segment == 4:
                embedding = 2
            elif position_in_segment == 5:
                embedding = 3
            elif position_in_segment == 6:
                embedding = 4
            elif position_in_segment == 7:
                embedding = 4
            
            x = segment_ID * 5 + embedding
        
        elif _super_layer == 5:
            unsqueezed_bitwidth = 9
            segment_ID = int(index_from_offset / unsqueezed_bitwidth)
            position_in_segment = index_from_offset % unsqueezed_bitwidth
        
            if position_in_segment == 0:
                embedding = 0
            elif position_in_segment == 1:
                embedding = 0
            elif position_in_segment == 2:
                embedding = 1
            elif position_in_segment == 3:
                embedding = 1
            elif position_in_segment == 4:
                embedding = 2
            elif position_in_segment == 5:
                embedding = 3
            elif position_in_segment == 6:
                embedding = 3
            elif position_in_segment == 7:
                embedding = 4
            elif position_in_segment == 8:
                embedding = 4
            x = segment_ID * 5 + embedding
        
        elif _super_layer == 6:
            unsqueezed_bitwidth = 10
            segment_ID = int(index_from_offset / unsqueezed_bitwidth)
            position_in_segment = index_from_offset % unsqueezed_bitwidth
        
            if position_in_segment == 0:
                embedding = 0
            elif position_in_segment == 1:
                embedding = 0
            elif position_in_segment == 2:
                embedding = 1
            elif position_in_segment == 3:
                embedding = 1
            elif position_in_segment == 4:
                embedding = 2
            elif position_in_segment == 5:
                embedding = 2
            elif position_in_segment == 6:
                embedding = 3
            elif position_in_segment == 7:
                embedding = 3
            elif position_in_segment == 8:
                embedding = 4
            elif position_in_segment == 9:
                embedding = 4
            
            x = segment_ID * 5 + embedding
        
        elif _super_layer == 7:
            unsqueezed_bitwidth = 11
            segment_ID = int(index_from_offset / unsqueezed_bitwidth)
            position_in_segment = index_from_offset % unsqueezed_bitwidth
        
            if position_in_segment == 0:
                embedding = 0
            elif position_in_segment == 1:
                embedding = 0
            elif position_in_segment == 2:
                embedding = 1
            elif position_in_segment == 3:
                embedding = 1
            elif position_in_segment == 4:
                embedding = 2
            elif position_in_segment == 5:
                embedding = 2
            elif position_in_segment == 6:
                embedding = 2
            elif position_in_segment == 7:
                embedding = 3
            elif position_in_segment == 8:
                embedding = 3
            elif position_in_segment == 9:
                embedding = 4
            elif position_in_segment == 10:
                embedding = 4           
            x = segment_ID * 5 + embedding       
        elif _super_layer == 8:       
            unsqueezed_bitwidth = 12
            segment_ID = int(index_from_offset / unsqueezed_bitwidth)
            position_in_segment = index_from_offset % unsqueezed_bitwidth    
            if position_in_segment == 0:
                embedding = 0
            elif position_in_segment == 1:
                embedding = 0
            elif position_in_segment == 2:
                embedding = 1
            elif position_in_segment == 3:
                embedding = 1
            elif position_in_segment == 4:
                embedding = 1
            elif position_in_segment == 5:
                embedding = 2
            elif position_in_segment == 6:
                embedding = 2
            elif position_in_segment == 7:
                embedding = 3
            elif position_in_segment == 8:
                embedding = 3
            elif position_in_segment == 9:
                embedding = 3
            elif position_in_segment == 10:
                embedding = 4
            elif position_in_segment == 11:
                embedding = 4
            
            x = segment_ID * 5 + embedding
    
        return ( x , y , embedding )
    
    def convert_one_frame(self,
                         this_frame # defined as 1d array with the size of 14336 
                         ):
        
        # Empty background
        converted_frame = np.zeros(shape=(56,160))

        for wireid in np.where(this_frame == 1)[0]:
            (pix_x,pix_y,_) = self.OR_Gate_mapping(self.Lining_dict[wireid][0],self.Lining_dict[wireid][1])
            converted_frame[pix_y,pix_x] = 1
        
        return converted_frame

    def convert_frame_batch(self,frame_batch):
        batch_size = frame_batch.shape[0]
        converted_frame_batch = np.zeros(shape=(batch_size,56,160))
        for frame_num in range(batch_size):
            converted_frame_batch[frame_num] = self.convert_one_frame(frame_batch[frame_num])
        return converted_frame_batch
    
    def create_system_verilog(self):
        code = ""
        
        # Logic to generate system verilog code
        # ports
        code += "module cdc_hit_image_converter( \n"
        ## PORT names
        code += "input wire CLK, \n"
        code += "input wire CDC_WIRE [14335:0], \n"
        code += "output reg [159:0] HIT_IMAGE_PIX [55:0] \n"        
        code += " );\n"

        #regs
        code += "reg cdc_wire_hit_status [14335:0]; \n"
        
        #syschrous on posedge clk                                   # PIPLINE STAGE 0
        code += "always@(posedge(CLK)) begin \n"
        code += "cdc_wire_hit_status <= CDC_WIRE;\n"
        #code += "end \n"
        #rewiring 
        cdc_to_pix_position = []
        for i in range(14336):
            (pix_x,pix_y,_) = self.OR_Gate_mapping(self.Lining_dict[i][0],self.Lining_dict[i][1])
            cdc_to_pix_position.append([pix_x,pix_y])
        #print(cdc_to_pix_position)
            
        for pos_y in range(56):
            for pos_x in range(160):
                code += f" HIT_IMAGE_PIX[{pos_y}][{pos_x}] <= "
                newline_flag = True
                for wire_id in range(14336):
                    if (cdc_to_pix_position[wire_id] == [pos_x,pos_y]):                       
                        code += f"cdc_wire_hit_status[{wire_id}]" if newline_flag else \
                        f"| cdc_wire_hit_status[{wire_id}]"                        
                        newline_flag = False
                code +=";\n"
        code += "end \n"
        code += "endmodule"

        return code 

    def create_instance(self):

        # wires for connection
        code = "\n"
        code += "wire [159:0] HIT_IMAGE_PIX [55:0];\n"
        #module instance
        code += "cdc_hit_image_converter cdc_hit_image_converter_ins (.*);     \n"

        return code


class FPGA_parallel_conv2D():
    def __init__(
                self,
                input_size_x,
                input_size_y,
                 filter_param,
                 filter_strides_x,
                 filter_strides_y,
                 adder_input_size,
                 pool_type,
                 pool_size,
                 pool_strides
                 ):
        # Input Layer 
        print(f"NEW SHIT JUST GOT MADE")

        self.in_x    = input_size_x
        self.in_y    = input_size_y

        # Conv2D_layer
        self.filter_x          = filter_param.shape[1]
        self.filter_y          = filter_param.shape[0]
        self.filter_param = tf.constant(filter_param.reshape(self.filter_y,self.filter_x,1,1), dtype = tf.float32)
        
        self.filter_strides_x = filter_strides_x
        self.filter_strides_y = filter_strides_y
        self.filter_strides = [1,filter_strides_y,filter_strides_x,1]

        # Pool_layer
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.pool_x    = pool_size[1]
        self.pool_y    = pool_size[0]
        
        self.pool_strides = pool_strides
        self.pool_strides_x = pool_strides[1]
        self.pool_strides_y = pool_strides[0]


        ### CONV 2D layer Valid check
        self.window_range_x = (self.in_x - self.filter_x) // self.filter_strides_x + 1      
        self.window_range_y = (self.in_y - self.filter_y) // self.filter_strides_y + 1
        
        self.valid_pix_x =   (self.window_range_x - 1) * self.filter_strides_x + self.filter_x 
        self.valid_pix_y =  (self.window_range_y - 1) *  self.filter_strides_y + self.filter_y 
        
        self.window_position_tensor = self._get_convwindow_position()
        self.window_num = self.window_position_tensor.shape[0]

        self._check_conv_missed_pix()

        ### Adder tree configuration

        self.adder_input_size = adder_input_size
        self.adder_tree_configuratoin_matrix = self._create_adder_configuration_matrix(self.adder_input_size)
        self.adder_tree_config = self.adder_tree_configuratoin_matrix
        self.adder_tree_pixs_range = self.filter_x * self.filter_y
        self.adder_tree_total_stage = len(self.adder_tree_config) - 1                                        
        self.overflow_digits_per_stage = self.adder_input_size - 1                          
        self.flattend_filter_parameter = self.filter_param.numpy().reshape(self.filter_x * self.filter_y) 
        self.max_weight = np.max(self.flattend_filter_parameter)
        self.mim_reg_width_for_parameter = int(math.log2(self.max_weight)) + 1
        self.conv_result_width = self.mim_reg_width_for_parameter + self.adder_tree_total_stage * self.overflow_digits_per_stage

        # get pool connection matrix
        self.pool_connection_matrix = self._create_pool_connection_matrix(self.adder_input_size)

        ### Pool Layer Valid check
        self.poolwindow_range_x = (self.window_range_x - self.pool_x ) // self.pool_strides_x + 1
        self.poolwindow_range_y = (self.window_range_y - self.pool_y ) // self.pool_strides_y + 1
        self.poolwindow_num = self.poolwindow_range_x * self.poolwindow_range_y

        ### calculate valid pix range on pool layer
        self.valid_pool_pix_x = (self.poolwindow_range_x - 1) * self.pool_strides_x + self.pool_x
        self.valid_pool_pix_y = (self.poolwindow_range_y - 1) * self.pool_strides_y + self.pool_y 

        self._check_pool_missed_pix()
        self.pool_window_position_tensor = self._get_poolwindow_position()
        #print(f"NEW SHIT generated !")

    def _check_conv_missed_pix(self):
        
        #print(f"There are {self.valid_pix_x} valid pix  on X directions")
        #print(f"There are {self.valid_pix_y} valid pix  on Y directions")
        print(f"The give input image shape is ({self.in_y} , {self.in_x} ) ")
        print(f"Valid pix range under this conv-configuration ({self.valid_pix_y},{self.valid_pix_x} )")
        print(f"Convolution window range is ({self.window_range_y},{self.window_range_x})")
        if(self.in_x - self.valid_pix_x == 0):
            print(f"-->Nice, All conv_pix in X direction is valid")
        else:
            #print(f"{self.in_x - self.valid_pix_x} pixels missed on X direction")
            pix_to_be_duplicat_x = self.filter_strides_x - (self.in_x - self.valid_pix_x)  
            print(f"Duplicat  {pix_to_be_duplicat_x} pix in X direction!")
        if(self.in_y - self.valid_pix_y == 0):
            print(f"-->Nice, All conv_pix on Y direction is Valid")
        else:
            #print(f"{self.in_y - self.valid_pix_y} pixels missed on Y direction")
            pix_to_be_duplicat_y = self.filter_strides_y - (self.in_y - self.valid_pix_y)
            print(f"Duplicat {pix_to_be_duplicat_y} pix in Y direction!")

    def _check_pool_missed_pix(self):

        print(f"Valid pix range under this pool-configuration ({self.valid_pool_pix_y,self.valid_pool_pix_x})")
        print(f" There are {self.poolwindow_range_x} valid pool window on X direction")
        print(f" There are {self.poolwindow_range_y} valid pool window on Y direction")
        #print(f"Unvalid pix ({self.} )")

    def forward_calculation(self,x):
        x = tf.nn.conv2d(
        x,
        self.filter_param,
        self.filter_strides,
        "VALID")
        if self.pool_type == "max_pool":
            y = tf.nn.max_pool2d(
            x,
            ksize=    self.pool_size,
            strides=  self.pool_strides,
            padding = "VALID")
        elif self.pool_type == "average_pool":
            y = tf.nn.avg_pool2d(
            x,
            ksize=    self.pool_size,
            strides=  self.pool_strides,
            padding = "VALID")
        else:
            raise TypeError(f"The type you give is unvalid: please either give max_pool or average_pool")       
        return y
    
    def trace_conv2d_layer(self,x):
        x = tf.nn.conv2d(
        x,
        self.filter_param,
        self.filter_strides,
        "VALID")
        return x
    
    def _get_convwindow_position(self):
        
        window_selection_list = []

        for window_num_y in range(self.window_range_y):
            for window_num_x in range(self.window_range_x):
                pixs_in_this_window = []
                anker_x = 0 if window_num_x==0 else window_num_x * self.filter_strides_x 
                anker_y = 0 if window_num_y==0 else window_num_y * self.filter_strides_y 
                for pos_y_in_this_window in range(anker_y,    anker_y + self.filter_y):
                    for pos_x_in_this_window in range(anker_x,anker_x + self.filter_x):
                        pixs_in_this_window.append( [ pos_x_in_this_window ,pos_y_in_this_window ] )
                window_selection_list.append(pixs_in_this_window)

        return np.array(window_selection_list,dtype = object)
    
    def _create_adder_configuration_matrix(self,adder_input_size):

        """
        This Function return a 3D array with the shape 
        
        [Stage_idx, Node_num_in_this Stage, Node_connection from last stage]

        this function is intend to called by the function below for svcode generation
        
        """

        connection_matrix_per_stage = []                                                              # 3D array, indexed by stage
        connection_matrix_first_stage =[i for i in range(self.filter_x * self.filter_y)]
        connection_matrix_per_stage.append(connection_matrix_first_stage)
        
        # calculate the connection_matrix untill the outcome of this stage only takes 1 reg and then exit
        process = True
        anker   = 0                                                                                   # holds the n-th input index from last stage
        stage_in_process = 0

        while(process):
            
            # households for each calculation
            input_size_this_stage = len(connection_matrix_per_stage[stage_in_process])
            connection_matrix_in_this_stage = [] # holds the connection for this stage
            anker = 0
            
            # add nornaml connections in the connections matrix, if exits
            for which_reg in range(input_size_this_stage // adder_input_size): # Loop all valid normal connections
                connection_to_this_reg = []
                for connections in range(adder_input_size*which_reg,adder_input_size*(which_reg + 1)):
                    connection_to_this_reg.append(connections)
                    anker = connections
                connection_matrix_in_this_stage.append(connection_to_this_reg)
                
            # for residual connections
            if(input_size_this_stage % adder_input_size != 0):
                connection_to_this_reg = []
                if(input_size_this_stage < adder_input_size): # if only residual connections in this stage
                    for residual_reg in range(input_size_this_stage % adder_input_size):
                        connection_to_this_reg.append(anker + residual_reg)
                else:
                    for residual_reg in range(input_size_this_stage % adder_input_size):
                        connection_to_this_reg.append(anker + 1 + residual_reg)
                connection_matrix_in_this_stage.append(connection_to_this_reg)
            
            # save the current connection in the matrix
            connection_matrix_per_stage.append(connection_matrix_in_this_stage)
            stage_in_process += 1 
            
            if len(connection_matrix_per_stage[stage_in_process]) == 1:
                print(f"adder tree solution with current setting found! ")
                break
            if stage_in_process >= 99:
                raise ValueError(f"hy fuck you! adder tree stage passed 99, something is wrong!")
            
        return connection_matrix_per_stage
    def _get_poolwindow_position(self):
            pool_window_position_list = []
            for poolwindow_num_y in range(self.poolwindow_range_y):
                for poolwindow_num_x in range(self.poolwindow_range_x):
                    convwindow_num_in_this_poolwindow = []
                    ## anker is defined as the position of left upper corner
                    anker_x = 0 if poolwindow_num_x==0 else poolwindow_num_x * self.pool_strides_x
                    anker_y = 0 if poolwindow_num_y==0 else poolwindow_num_y * self.pool_strides_y
                    for convwindow_y_in_this_poolwindow in range(anker_y,anker_y+self.pool_y): 
                        for convwindow_x_in_this_poolwindow in range(anker_x,anker_x+self.pool_x):
                            convwindow_num_in_this_poolwindow.append([convwindow_x_in_this_poolwindow,convwindow_y_in_this_poolwindow])
                    pool_window_position_list.append(convwindow_num_in_this_poolwindow)
            return np.array(pool_window_position_list,dtype= object )
    
    def _create_pool_connection_matrix(self,adder_input_size):

        connection_matrix_per_stage = []                                                              # 3D array, indexed by stage
        connection_matrix_first_stage =[i for i in range(self.pool_x* self.pool_y)]
        connection_matrix_per_stage.append(connection_matrix_first_stage)
        
        # calculate the connection_matrix untill the outcome of this stage only takes 1 reg and then exit
        process = True
        anker   = 0                                                                                   # holds the n-th input index from last stage
        stage_in_process = 0

        while(process):
            
            # households for each calculation
            input_size_this_stage = len(connection_matrix_per_stage[stage_in_process])
            connection_matrix_in_this_stage = [] # holds the connection for this stage
            anker = 0
            
            # add nornaml connections in the connections matrix, if exits
            for which_reg in range(input_size_this_stage // adder_input_size): # Loop all valid normal connections
                connection_to_this_reg = []
                for connections in range(adder_input_size*which_reg,adder_input_size*(which_reg + 1)):
                    connection_to_this_reg.append(connections)
                    anker = connections
                connection_matrix_in_this_stage.append(connection_to_this_reg)
                
            # for residual connections
            if(input_size_this_stage % adder_input_size != 0):
                connection_to_this_reg = []
                if(input_size_this_stage < adder_input_size): # if only residual connections in this stage
                    for residual_reg in range(input_size_this_stage % adder_input_size):
                        connection_to_this_reg.append(anker + residual_reg)
                else:
                    for residual_reg in range(input_size_this_stage % adder_input_size):
                        connection_to_this_reg.append(anker + 1 + residual_reg)
                connection_matrix_in_this_stage.append(connection_to_this_reg)
            
            # save the current connection in the matrix
            connection_matrix_per_stage.append(connection_matrix_in_this_stage)
            stage_in_process += 1 
            
            if len(connection_matrix_per_stage[stage_in_process]) == 1:
                print(f"adder tree solution with current setting found! ")
                break
            if stage_in_process >= 99:
                raise ValueError(f"hy fuck you! adder tree stage passed 99, something is wrong!")
            
        return connection_matrix_per_stage
                    



##############################################################################################
#############################       Conv2D Window Spiliter       #############################
############################################################################################## 


    def create_systemverilog_convwindow_spliter(self):
        
        """
        This Function returns the SystemVerilog code for the Convwindow spliter module as String
        !!!!    Left up corner is considered as LSB
        !!!!    Right Down corner is considered as MSB

        """    

        code = ""
        ports = ""             
        ports += f"module conv2d_window_spliter(             \n"
        ports += f"input wire [{self.in_x - 1 }:0] HIT_IMAGE_PIX[ {self.in_y -1 }:0],   \n"
        ports += f"output wire[{self.filter_x * self.filter_y - 1}:0] CONV_WINDOW[{self.window_num - 1}:0]  \n"
        ports += f");"
        code += ports

        # Connect wires for each window
        for windowID,connected_pixs in enumerate(self.window_position_tensor):
            code += f"assign CONV_WINDOW[{windowID}] = "
            code += " { "
            for _,[pix_x,pix_y] in enumerate(np.flip(connected_pixs,axis = 0)):
                code += f"HIT_IMAGE_PIX[{pix_y}][{pix_x}]" if _ ==0 else f", HIT_IMAGE_PIX[{pix_y}][{pix_x}] "
            code += " }; \n  "
        code += "endmodule"

        return code


    def create_instance_convwindow_spliter(self):
        """
        This Funrions returns the Systemverilog code for instanziation the convwindow spliter in the top modul
        """
        # connections wires
        code = ""
        code += f"// Connection Wires                                       \n "
        code += f" wire [{self.filter_x * self.filter_y - 1}:0] CONV_WINDOW[{self.window_num - 1}:0] ; \n "
        code += f"// Module ins and port map                                \n "
        # port map
        code += f"conv2d_window_spliter conv2d_window_spliter_ins  (        \n "
        code += f".HIT_IMAGE_PIX(HIT_IMAGE_PIX),                            \n "
        code += f".CONV_WINDOW(CONV_WINDOW)                                 \n "
        code += f" );                                                       \n "
        return code
    
##############################################################################################
#############################       Conv2D Adder Treeeeeeeee       ###########################
##############################################################################################    

    def create_systemverilog_adder_tree(self):

        """
        This Function returns the hardware code for adder tree module as string 

        """
        
        # Create the connection matrix
        adder_tree_config = self.adder_tree_configuratoin_matrix
        
        # ports
        ports =  f"module conv2d_adder_tree(                 \n "
        ports += f"input wire CLK,                           \n "
        ports += f"input wire [{self.adder_tree_pixs_range - 1}:0] PIX,      \n "
        ports += f"output wire[{self.conv_result_width - 1}:0] CONV_RESULT  \n "
        ports += f");\n "

        # parameter (weights for each pix)
        parameters = f" // parameters (weights for each PIX)  \n "
        for idx,weights in enumerate(self.flattend_filter_parameter.astype(int)):
            parameters += f"parameter W_{idx} = {weights};   \n" 
        
        # regs
        regs = f" // regs for each partial sum stage         \n "
        for stage_idx, connections in enumerate(adder_tree_config[1:]):
            reg_width_for_this_stage = self.mim_reg_width_for_parameter + (stage_idx + 1) * self.overflow_digits_per_stage 
            regs += f"reg [{reg_width_for_this_stage -1}:0] S_{stage_idx} [{len(connections) -1 }:0];\n " if stage_idx < self.adder_tree_total_stage - 1\
              else f"reg [{reg_width_for_this_stage -1}:0] S_{stage_idx} ;\n "

        # logic
        logics = f"always@(posedge(CLK)) begin \n "
        for stage_idx, stage_connections in enumerate(adder_tree_config[1:]):
            logics += f" // Stage{stage_idx} \n "
            for node_idx, node_connections in enumerate(stage_connections):
                logics += f"  S_{stage_idx}[{node_idx}] <= " if stage_idx < self.adder_tree_total_stage -1\
                else  f"  S_{stage_idx} <= "
                new_line_flag = True
                for _ , node_from_previous_stage in enumerate(node_connections):
                    if stage_idx == 0:
                        logics +=  f" W_{node_from_previous_stage} * PIX[{node_from_previous_stage}]" if new_line_flag else\
                                   f" + W_{node_from_previous_stage} * PIX[{node_from_previous_stage}]"  
                        new_line_flag = False
                    else:
                        logics +=  f" S_{stage_idx-1}[{node_from_previous_stage}]   " if new_line_flag else\
                                   f" + S_{stage_idx-1}[{node_from_previous_stage}]"
                        new_line_flag = False
                logics += f"; \n "
        logics += f"end\n "

        # code Assembly
        code = ""
        code = code + ports + parameters + regs + logics 

        #result assignment
        code += f"assign CONV_RESULT = S_{self.adder_tree_total_stage - 1} ;\n "
        code += f"endmodule"
        return code
    

    def create_instance_adder_tree(self):

        code = ""
        code += "// wires for connection                                   \n"
        code += f"wire[{self.conv_result_width - 1}:0] CONV_RESULT [{self.window_num - 1}:0];                  \n"
        code += f"genvar i ;                                               \n"
        code += f"generate                                                 \n "
        code += f"for(i = 0 ; i < {self.window_num} ; i = i + 1 ) begin    \n"
        code += f" conv2d_adder_tree conv2d_adder_tree_ins(                \n  "
        code += f".CLK(CLK),                                               \n"
        code += f".PIX(CONV_WINDOW[i]),                                    \n"
        code += f".CONV_RESULT(CONV_RESULT[i]) "
        code += f");                                                       \n"
        code += f"end                                                      \n "
        code += f"endgenerate                                              \n" 
        return code

##############################################################################################
#############################          Pool Window Spliter         ###########################
##############################################################################################


    def create_systemverilog_poolwindow_spliter(self):

        ## ports 
        code = ""
        code += f" module poolwindow_spliter ( \n"
        code += f" input wire [ {self.conv_result_width - 1} : 0] CONV_RESULT [ {self.window_num - 1}:0],  \n"
        code += f" output wire [{self.pool_x*self.pool_y-1}:0] [{self.conv_result_width - 1} : 0] POOL_WINDOW [{self.poolwindow_num -1}:0 ]  \n"
        code += f");                           \n"
        
        for poolwindowID,connected_conv_window in enumerate(self.pool_window_position_tensor):
            code += f"assign POOL_WINDOW [{poolwindowID}]  =  "
            code += " {  "
            newline_flag = True
            for _,[_convwindow_x,_convwindow_y] in enumerate(np.flip(connected_conv_window,axis=0)):
                code += f"CONV_RESULT [ {_convwindow_y*( self.window_range_x  ) + _convwindow_x}] " if _==0 else \
                f",CONV_RESULT [ {_convwindow_y*( self.window_range_x ) + _convwindow_x}] "
            code +=" };  \n"
        code += f"endmodule"
        return code
    


    def create_instance_poolwindow_spliter(self):
        # wires
        code = ""
        code +=f" wire [{self.pool_x*self.pool_y-1}:0] [{self.conv_result_width - 1} : 0] POOL_WINDOW [{self.poolwindow_num -1}:0 ] ; \n"
        # pool_window_spliter
        code +=f"poolwindow_spliter poolwindow_spliter_ins ( \n"
        code +=f".CONV_RESULT(CONV_RESULT), \n"
        code +=f".POOL_WINDOW(POOL_WINDOW)  \n"
        code +=f" ); \n"
        return code 
##############################################################################################
#############################          Pool Layer             ###########################
##############################################################################################
    
    def create_systemverilog_pool2D(self):
        """
        This Function returns the hardware code for adder tree module as string 

        """
        
        # Create the connection matrix
        adder_tree_config = self.pool_connection_matrix
        
        # ports
        if self.pool_type == "average_pool":

            ports =  f"module average_pool(                 \n "
        elif self.pool_type == "max_pool":
            ports =  f"module max_pool(                      \n "
        ports += f"input wire CLK,                           \n "
        ports += f"input wire [ {self.pool_x *self.pool_y -1}:0 ] [{self.conv_result_width - 1}:0] POOL_WINDOW,      \n "
        ports += f"output wire[15:0] POOL_RESULT    \n "
        ports += f");\n "
      
        # regs
        regs = f" // regs for each partial sum stage         \n "
        for stage_idx, connections in enumerate(adder_tree_config[1:]):
            reg_width_for_this_stage = self.conv_result_width
            regs += f"reg [{reg_width_for_this_stage -1}:0] S_{stage_idx} [{len(connections) -1 }:0];\n " if stage_idx < self.adder_tree_total_stage - 1\
              else f"reg [{reg_width_for_this_stage -1}:0] S_{stage_idx} ;\n "

        # logic
        if self.pool_type == "average_pool":
            logics = f"always@(posedge(CLK)) begin \n "
            for stage_idx, stage_connections in enumerate(adder_tree_config[1:]):
                logics += f" // Stage{stage_idx} \n "
                for node_idx, node_connections in enumerate(stage_connections):
                    logics += f"  S_{stage_idx}[{node_idx}] <= " if stage_idx < self.adder_tree_total_stage -1\
                    else  f"  S_{stage_idx} <= "
                    new_line_flag = True
                    for _ , node_from_previous_stage in enumerate(node_connections):
                        if stage_idx == 0:
                            logics +=  f"  POOL_WINDOW[{node_from_previous_stage}]" if new_line_flag else\
                                    f" +  POOL_WINDOW[{node_from_previous_stage}]"  
                            new_line_flag = False
                        else:
                            logics +=  f" S_{stage_idx-1}[{node_from_previous_stage}]   " if new_line_flag else\
                                    f" + S_{stage_idx-1}[{node_from_previous_stage}]"
                            new_line_flag = False
                    logics += f"; \n "
            logics += f"end\n "

        elif self.pool_type == "max_pool":
            logics = f"always@(posedge(CLK)) begin \n "
            for stage_idx, stage_connections in enumerate(adder_tree_config[1:]):
                logics += f" // Stage{stage_idx} \n "
                for node_idx, node_connections in enumerate(stage_connections):
                    logics += f"  S_{stage_idx}[{node_idx}] <= " if stage_idx < self.adder_tree_total_stage -1\
                    else  f"  S_{stage_idx} <= "
                    new_line_flag = True
                    for _ , node_from_previous_stage in enumerate(node_connections):
                        if stage_idx == 0:
                            logics +=  f" (POOL_WINDOW[ {node_from_previous_stage}] > POOL_WINDOW[ {node_from_previous_stage+1}])?" if new_line_flag else\
                                    f" POOL_WINDOW[ {node_from_previous_stage-1}]:POOL_WINDOW[ {node_from_previous_stage}] "  
                            new_line_flag = False
                        else:
                            logics +=  f" (S_{stage_idx-1}[{node_from_previous_stage}]  > S_{stage_idx-1}[{node_from_previous_stage+1}])?  " if new_line_flag else\
                                    f" S_{stage_idx-1}[{node_from_previous_stage-1}]: S_{stage_idx-1}[{node_from_previous_stage}]"
                            new_line_flag = False
                    logics += f"; \n "
            logics += f"end\n "
        else:
            raise TypeError(f"hy fuck you! give either max_pool or average_pool")

        # code Assembly
        code = ""
        code = code + ports + regs + logics 

        #result assignment
        code += f"// TODO  \n"
        code += f"//  aglinign to the correct ap<16,8> format  "
        code += f"assign POOL_RESULT = S_{len(adder_tree_config) - 2} ;\n "
        code += f"endmodule"
        return code
    def create_instance_pool2D(self):      
        code = ""
        #code += f"wire[{self.pool_x * self.pool_y - 1 }:0][{self.conv_result_width-1}:0]POOL_WINDOW [{self.poolwindow_num - 1}:0]  ; "
        code += f"wire[15:0] POOL_RESULT [{self.poolwindow_num -1}:0 ]   ;"
        code += f"\n"
        code += f"genvar j;\n"
        code += f"generate \n"
        code += f"for(j = 0 ; j < {self.poolwindow_num};j = j + 1 ) begin\n"
        if self.pool_type == "max_pool":
            code += f"max_pool max_pool_ins( \n"
        elif self.pool_type == "average_pool":
            code += f"average_pool average_pool_ins( \n"
        code += f".CLK(CLK),    \n"
        code += f".POOL_WINDOW(POOL_WINDOW[j]),\n"
        code += f".POOL_RESULT(POOL_RESULT[j])\n" 
        code += f" ); \n"
        code += f"end\n"
        code += f"endgenerate\n"

        code += f"wire [{16*self.poolwindow_num -1} :0] POOL_RESULT_FLATTED;\n"
        code += f"genvar k;           \n"
        code += f"generate            \n"
        code += f"for(k = 0; k <{self.poolwindow_num}; k = k + 1) begin \n"
        code += f"assign POOL_RESULT_FLATTED[k*16+15:k*16] = POOL_RESULT[k];\n"
        code += f"end  \n"
        code += f"endgenerate\n"
        return code 




class FPGA_hardware_handler():
    def __init__(self,FPGA_hit_image_converter_ins,FPGA_parral_conv2D_ins):
        self.hit_image_converter = FPGA_hit_image_converter_ins
        self.conv2d              = FPGA_parral_conv2D_ins
    def write_setting(self):
        setting = ""
        setting += f"-----INPUT SHAPE-----\n({self.conv2d.in_y},{self.conv2d.in_x})\n"
        setting += f"-----CONV_SETTING-----\nstrides:({self.conv2d.filter_strides_y},{self.conv2d.filter_strides_x})\n"
        setting += f"-----POOL_SETTING-----\ntype:{self.conv2d.pool_type}\npool_size({self.conv2d.pool_y},{self.conv2d.pool_x})\npool_strides({self.conv2d.pool_strides_y},{self.conv2d.pool_strides_x})"
        return setting
    # Forward Calculation
    def convert_to_FlattendDataset(self,x):
        x = self.trace_Pooled_IMAGE(x)
        x = x.reshape(x.shape[0],x.shape[1]*x.shape[2])
        return x
        
    def trace_HIT_IMAGE(self,x):
        x = self.hit_image_converter.convert_frame_batch(x)
        return x
    
    def trace_CONVED_IMAGE(self,x):

        x = self.trace_HIT_IMAGE(x)
        x = x.reshape(x.shape[0],x.shape[1],x.shape[2], 1)
        x = self.conv2d.trace_conv2d_layer(x)

        return x.numpy()


    def trace_Pooled_IMAGE(self,x):
        x = self.trace_HIT_IMAGE(x)
        x = x.reshape(x.shape[0],x.shape[1],x.shape[2], 1)
        x = self.conv2d.forward_calculation(x)
        return x.numpy()    


    # Hardware Code generation

    ## Hit_image_convertere
    def writefile_hit_image_converter(self,outdir):
        filename = "FPGA_hit_image_converter.sv"
        filepath = outdir + filename
        code = self.hit_image_converter.create_system_verilog()
        with open(filepath,"w") as file:
            file.write(code)
              
    ## Parallel window spliter
    def writefile_conv2d_window_spliter(self,outdir): 
        filename = "FPGA_conv2d_window_spiliter.sv"
        filepath = outdir + filename
        code = self.conv2d.create_systemverilog_convwindow_spliter()
        with open(filepath,"w") as file:
            file.write(code)
        
    ## Parallel Conv2D
    def writefile_parallel_conv2d(self,outdir):
        filename = "FPGA_conv2d.sv"
        filepath = outdir + filename
        code = self.conv2d.create_systemverilog_adder_tree()
        with open(filepath,"w") as file:
            file.write(code)

    ## Pool window selector
    def writefile_pool_window_spliter(self,outdir):
        filename = "FPGA_poolwindow_spliter.sv"
        filepath = outdir + filename
        code = self.conv2d.create_systemverilog_poolwindow_spliter()
        with open(filepath,"w") as file:
            file.write(code)

    ## pool Layer
    def writefile_parallel_pool(self,outdir):
        filename = "FPGA_pool2d.sv"
        filepath = outdir + filename
        code = self.conv2d.create_systemverilog_pool2D()
        with open(filepath,"w") as file:
            file.write(code)

    ## top module
    def writefile_top_module(self,outdir):
        
        filename = "top_module_before_NN.sv"
        filepath = outdir + filename

        code = "   "
        code += f"module top_module_before_NN(     \n "
        code += f"input wire CLK,                  \n "
        code += f"input wire CDC_WIRE [14335:0],   \n "
        code += f"output wire [{16*self.conv2d.poolwindow_num -1} :0] POOL_RESULT_FLATTED_TOP "
        code += f" );                                \n "
        code += self.hit_image_converter.create_instance()
        code += self.conv2d.create_instance_convwindow_spliter()
        code += self.conv2d.create_instance_adder_tree()
        code += self.conv2d.create_instance_poolwindow_spliter()
        code += self.conv2d.create_instance_pool2D()

        code += f"\n "
        code += f"assign POOL_RESULT_FLATTED_TOP = POOL_RESULT_FLATTED;"
        code += f"endmodule"
        with open(filepath,"w") as file:
            file.write(code )
        
    def generate_hardware_module(self,outdir):
        self.writefile_hit_image_converter(outdir)
        self.writefile_conv2d_window_spliter(outdir)
        self.writefile_parallel_conv2d(outdir)
        self.writefile_pool_window_spliter(outdir)
        self.writefile_parallel_pool(outdir)
        self.writefile_top_module(outdir)




""" 
--------------------------------------------
------------ old tramission map-------------
--------------------------------------------
    Hits_trans_map = np.array([
    # SuperLayer 0
    [0,1,2,3,4], # 0
    [0,1,2,3,4], # 1
    [0,1,2,3,4], # 2
    [0,1,2,3,4], # 3
    [0,1,2,3,4], # 4 
    [0,1,2,3,4], # 5
    [0,1,2,3,4], # 6
    [0,1,2,3,4], # 7
    
    # SuperLayer 1
    [0,1,2,3,4], # 8
    [0,1,2,3,4], # 9
    [0,1,2,3,4], # 10
    [0,1,2,3,4], # 11
    [0,1,2,3,4], # 12 
    [0,1,2,3,4], # 13
    
    # SuperLayer 2
    [0,0,1,2,3,4], # 14
    [0,1,1,2,3,4], # 15
    [0,1,2,2,3,4], # 16
    [0,1,2,3,3,4], # 17
    [0,1,2,3,4,4], # 18 
    [0,0,1,2,3,4], # 19
    
    # SuperLayer 3
    [0,1,1,2,3,3,4], # 20
    [0,1,2,2,3,4,4], # 21
    [0,0,1,2,3,3,4], # 22
    [0,1,1,2,3,4,4], # 23
    [0,0,1,2,2,3,4], # 24 
    [0,1,1,2,3,3,4], # 25
    
    # SuperLayer 4
    [0,0,1,2,2,3,4,4], # 26
    [0,0,1,1,2,3,3,4], # 27
    [0,1,1,2,2,3,4,4], # 28
    [0,0,1,2,2,3,3,4], # 29
    [0,1,1,2,3,3,4,4], # 30
    [0,0,1,2,2,3,4,4],  # 31
    
    # SuperLayer 5
    [0,1,1,2,2,3,3,4,4], # 32
    [0,0,1,2,2,3,3,4,4], # 33
    [0,0,1,1,2,3,3,4,4], # 34
    [0,0,1,1,2,2,3,4,4], # 35
    [0,0,1,1,2,2,3,3,4], # 36 
    [0,1,1,2,2,3,3,4,4],  # 37
    
    # SuperLayer 6
    [0,0,1,1,2,2,3,3,4,4], # 38
    [0,0,1,1,2,2,3,3,4,4], # 39
    [0,0,1,1,2,2,3,3,4,4], # 40
    [0,0,1,1,2,2,3,3,4,4], # 41
    [0,0,1,1,2,2,3,3,4,4], # 42
    [0,0,1,1,2,2,3,3,4,4],  # 43
    
    # SuperLayer 7
    [0,0,0,1,1,2,2,3,3,4,4], # 44
    [0,0,1,1,1,2,2,3,3,4,4], # 45
    [0,0,1,1,2,2,2,3,3,4,4], # 46
    [0,0,1,1,2,2,3,3,3,4,4], # 47
    [0,0,1,1,2,2,3,3,4,4,4], # 48 
    [0,0,0,1,1,2,2,3,3,4,4],  # 49
    
    # SuperLayer 8
    [0,0,1,1,1,2,2,3,3,3,4,4], # 50
    [0,0,1,1,2,2,2,3,3,4,4,4], # 51
    [0,0,0,1,1,2,2,3,3,3,4,4], # 52
    [0,0,1,1,1,2,2,3,3,4,4,4], # 53
    [0,0,0,1,1,2,2,2,3,3,4,4], # 54 
    [0,0,1,1,1,2,2,3,3,3,4,4]  # 55
    ],dtype = object)
    """



































"""
-----------------------function test---------------------- 



print(f"---------------------------")
print(f"There goes the test code!!")
print(f"---------------------------")

import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v2 as tf



kernel = np.ones(shape=(7,7))
print(kernel)

ix = 160
iy = 56
FPGA_conv_kernel = FPGA_parallel_conv2D(
input_size_x= ix,
input_size_y= iy ,
filter_param   = kernel,
filter_strides_x = 4 ,
filter_strides_y = 4,
pool_type      = "max_pool",
#pool_type      = "average_pool",    
pool_size      = (2,2),
pool_strides   = (2,2),
adder_input_size=2
)



#Background 
print(f"CONV WINDOW POSITION CHECK")
background = np.zeros(shape=(iy,ix))

window_num = 200
window_position_matrix = FPGA_conv_kernel.window_position_tensor
for idx,[x,y] in enumerate(window_position_matrix[window_num]):
    background[y,x] = 1
print(f"input image shape is {background.shape}")
#plt.show()



# Dummy Input
background = background.reshape(1,iy,ix,1)

# CONV2D result
conv_tester_out = FPGA_conv_kernel.trace_conv2d_layer(background)
#tester_out = tester_out.reshape(tester_out.shape[1],tester_out.shape[2])
print(f"shape after conv is {conv_tester_out.shape}")

# Polled Layer Result
pooled_tester_out = FPGA_conv_kernel.forward_calculation(background)
print(f"shape after pooled is {pooled_tester_out.shape}")

#plt.imshow(tester_out)
####
print("Parameter test")
print(f"{FPGA_conv_kernel.window_range_x}")
print(f"{FPGA_conv_kernel.window_range_y}")

#print(tester_out)

print(f"---------------------------------")
print(f"pool window position tensor")
print(FPGA_conv_kernel.pool_window_position_tensor)
print(FPGA_conv_kernel.pool_window_position_tensor.shape)

testcode = FPGA_conv_kernel.create_systemverilog_poolwindow_spliter()
with open("fuckmeintheass.sv","w") as file:
    file.write(testcode)


sigma = 2.0
ksize = 7
kernel = np.zeros((ksize, ksize))
def gaussian_kernel_2d(x, y, sigma):
    return np.exp(-(x**2 + y**2) / (2 * sigma**2))
center = np.array(kernel.shape) // 2

for x in range(kernel.shape[0]):
    for y in range(kernel.shape[1]):
        kernel[x, y] = gaussian_kernel_2d(x-center[0], y-center[1], sigma)

# scale the kernel to our maximum precision so we can quantize it. The highest value will always be in the center.
resolution = 2**4  # 8-bit resolution
kernel *= resolution / kernel.max()

# integer quantization
kernel = kernel.astype(np.int32)
print(kernel)

#plt.imshow(kernel, cmap='gray')
#plt.show()
print(f"---------------------------")
print(f"--INSTANCE new hardware ---")
print(f"---------------------------")

FPGA_converter = FPGA_hit_image_converter()

ix = 160
iy = 56

FPGA_conv_kernel = FPGA_parallel_conv2D(
input_size_x= ix,
input_size_y= iy ,
filter_param   = kernel,
#filter_strides = [1,4,4,1], ## Defined as [in channel, strides_y , strides_x , out_channel]
#pool_type      = "max_pool",
pool_type      = "average_pool",    
pool_size      = (2,2),
pool_strides   = (2,2),
filter_strides_x = 4 ,
filter_strides_y = 4,
adder_input_size = 2
)
my_hardware_module = FPGA_hardware_handler(FPGA_converter,FPGA_conv_kernel)
#print(FPGA_conv_kernel.pool_window_position_tensor)
#print(FPGA_conv_kernel.window_range_x)
#my_hardware_module.generate_hardware_module("myshit/")

dummy_dataset = np.ones((10,14336))
dummy_hit_image = my_hardware_module.trace_HIT_IMAGE(dummy_dataset)
print(dummy_hit_image.shape)
dummy_conved_image = my_hardware_module.trace_CONVED_IMAGE(dummy_dataset)
print(dummy_conved_image.shape)
dummy_pooled_image = my_hardware_module.trace_Pooled_IMAGE(dummy_dataset)
print(dummy_pooled_image.shape)
dummy_converted_dataset = my_hardware_module.convert_to_FlattendDataset(dummy_dataset)
print(dummy_converted_dataset.shape)


"""