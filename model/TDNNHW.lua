-- Time-delayed Neural Network (i.e. 1-d CNN) with multiple filter widths

local TDNNHW = {}
--local cudnn_status, cudnn = pcall(require, 'cudnn')

LookupTable = nn.LookupTable
print('use nn.LookupTable')

function TDNNHW.tdnnhw(max_length, input_size, feature_maps, kernels, nCharVocab, nCharEmbed, HWsize, L_HW, bias, f)
  -- length = length of sentences/words (zero padded to be of same length)
  -- input_size = embedding_size
  -- feature_maps = table of feature maps (for each kernel width)
  -- kernels = table of kernel widths
  local layer1_concat, tdnnoutput
  char_vec_layer = LookupTable(nCharVocab, nCharEmbed)   char_vec_layer.name = 'char_vecs'

  local input = nn.Identity()() --input is batch_size x length x input_size
  local input2 = char_vec_layer(input)

  local layer1 = {}
  for i = 1, #kernels do  -- Compute for every different kenrel type? that would be inefficient
    local reduced_l = max_length - kernels[i] + 1
    local pool_layer
    if opt.cudnn == 1 then
      -- Use CuDNN for temporal convolution.
      if not cudnn then require 'cudnn' end
      -- Fake the spatial convolution.
      local conv = cudnn.SpatialConvolution(1, feature_maps[i], input_size, kernels[i], 1, 1, 0)
      conv.name = 'conv_filter_' .. kernels[i] .. '_' .. feature_maps[i]
      -- local conv_layer = conv(nn.View(1, -1, input_size):setNumInputDims(2)(input))
      local conv_layer = conv(nn.View(1, -1, input_size):setNumInputDims(2)(input2))
      pool_layer = nn.Squeeze()(cudnn.SpatialMaxPooling(1, reduced_l, 1, 1, 0, 0)(nn.Tanh()(conv_layer))) -- Slightly modified to support online learning : See utils/Squeeze.lua
    else
      -- Temporal conv. much slower
      local conv = nn.TemporalConvolution(input_size, feature_maps[i], kernels[i])
      local conv_layer = conv(input)
      conv.name = 'conv_filter_' .. kernels[i] .. '_' .. feature_maps[i]
      --pool_layer = nn.Max(2)(nn.Tanh()(conv_layer))
      pool_layer = nn.TemporalMaxPooling(reduced_l)(nn.Tanh()(conv_layer))
      pool_layer = nn.Squeeze()(pool_layer)  -- Slightly modified to support online learning : See utils/Squeeze.lua
    end
    table.insert(layer1, pool_layer)
  end
  if #kernels > 1 then
    layer1_concat = nn.JoinTable(2)(layer1)
    tdnnoutput= layer1_concat
  else
    tdnnoutput = layer1[1]
  end

  -- size = dimensionality of inputs
  -- num_layers = number of hidden layers (default = 1)
  -- bias = bias for transform gate (default = -2)
  -- f = non-linearity (default = ReLU)

  if L_HW >=1 then
  local transform_gate, carry_gate
  local L_HW = L_HW or 1
  local bias = bias or -2
  local f = f or nn.ReLU()
  local inputs = {[1]=tdnnoutput}
  for i = 1, L_HW do
    output_module = f(nn.Linear(HWsize, HWsize)(inputs[i]))
    transform_gate = nn.Sigmoid()(nn.AddConstant(bias)(nn.Linear(HWsize, HWsize)(inputs[i])))
    carry_gate = nn.AddConstant(1)(nn.MulConstant(-1)(transform_gate))
    output_module = nn.CAddTable()({
      nn.CMulTable()({transform_gate, output_module}),
      nn.CMulTable()({carry_gate, inputs[i]}) })
    table.insert(inputs, output_module)
  end
  else
        output_module = tdnnoutput
  end
  return nn.gModule({input}, {output_module})
end

return TDNNHW
