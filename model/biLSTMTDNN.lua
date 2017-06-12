require 'nngraph'


local biLSTMTDNN = {}


local ok, cunn = pcall(require, 'fbcunn')
if not ok then
  LookupTable = nn.LookupTable
  print('use nn.LookupTable')
else
  LookupTable = nn.LookupTableGPU
  print('use fbcunns LookupTable')
end



function biLSTMTDNN.lstmtdnn(rnn_size,nRNNLayer, dropout, nWordVocab, nWordEmbed,nCharVocab,nCharEmbed,
  feature_maps, kernels, max_length, use_char,use_word, nHWLayer)

  -- rnn_size = dimensionality of LSTM block size
  -- nRNNLayer = #RNN layers
  -- dropout = dropout probability (default = 0)
  -- nWordVocab = #Vocabulary in word dictionary (this may not be used when we build wordvec from character) ==> only used when use_words = 1
  -- nWordEmbed = Word embedding size (default = 100) ==> only used when use_words = 1
  -- nCharVocab = #Vocabulary in char dictionary (default = 32)
  -- nCharEmbed = Char embedding size
  -- feature_maps = table of feature map sizes for each kernel width
  -- kernels = table of kernel widths
  -- max_length = max length of a word
  -- use_char = word vector from C2W
  -- use_word = word vector from random initialization
  -- nHWLayer = #Highway MLP layers

  dropout = dropout or 0

  -- there will be 4*nLayer+2 inputs
  local char_vec_layer, word_vec_layer, x, input_size_L, word_vec, char_vec
  local nHWLayer = nHWLayer or 0
  local inputs = {}    -- sets of activation is saved in here


  -- C2W
  if opt.use_chars == 1 then
    -- input character  (batch_sz x max_word_length) --> for 1st trial, we will provide code with batch_sz = 1
    table.insert(inputs, nn.Identity()())  -- x_f
    table.insert(inputs, nn.Identity()())  -- x_b
    print('nCharVocab = '..nCharVocab)
    print('nCharEmbed = '..nCharEmbed)
    char_vec_layer_f = LookupTable(nCharVocab, nCharEmbed)
    char_vec_layer_b = LookupTable(nCharVocab, nCharEmbed)
    char_vec_layer_f.name = 'char_vecs_f'  char_vec_layer_b.name = 'char_vecs_b'

  end
  if opt.use_words == 1 then
    -- input word (batch_sz x 1)  --> for 1st trial, we will provide code with batch_sz = 1
    table.insert(inputs, nn.Identity()())  -- x_f
    table.insert(inputs, nn.Identity()())  -- x_b
    word_vec_layer_f = LookupTable(nWordVocab, nWordEmbed)
    word_vec_layer_b = LookupTable(nWordVocab, nWordEmbed)
    word_vec_layer_f.name = 'word_vecs_f'  word_vec_layer_b.name = 'word_vecs_b'
  end

  for L = 1,nRNNLayer do
    table.insert(inputs, nn.Identity()()) -- prev_c_forward[L]
    table.insert(inputs, nn.Identity()()) -- prev_c_backward[L]
    table.insert(inputs, nn.Identity()()) -- prev_h_forward[L]
    table.insert(inputs, nn.Identity()()) -- prev_h_backward[L]
  end

  local outputs = {}
  for L = 1,nRNNLayer do

    -- the input to this layer
    if L == 1 then
      if opt.use_chars == 1 then
        
        char_vec_f = char_vec_layer_f(inputs[1])
        char_vec_b = char_vec_layer_b(inputs[2])
        
        --[[
        char_vec_f = char_vec_layer_f(nn.Identity()(inputs[1]))
        char_vec_b = char_vec_layer_b(nn.Identity()(inputs[2]))
        --]]
        
        local char_tdnn_f = TDNN.tdnn(max_length, nCharEmbed, feature_maps, kernels)
        local char_tdnn_b = TDNN.tdnn(max_length, nCharEmbed, feature_maps, kernels)
        char_tdnn_f.name = 'tdnn_f'  char_tdnn_b.name = 'tdnn_b'

        local tdnn_output_f = char_tdnn_f(char_vec_f)
        local tdnn_output_b = char_tdnn_b(char_vec_b)
        -- local tdnn_output = nn.JoinTable(2)({tdnn_output_f,tdnn_output_b})
        input_size_L = torch.Tensor(feature_maps):sum()
        if opt.use_words == 1 then
          word_vec_f = word_vec_layer(inputs[3])
          word_vec_b = word_vec_layer(inputs[4])
          x_f = nn.CAddTable()({char_vec_f, word_vec_f}) -- x_f = nn.JoinTable(2)({char_vec_f, word_vec_f})
          x_b = nn.CAddTable()({char_vec_b, word_vec_b}) -- x_b = nn.JoinTable(2)({char_vec_b, word_vec_b})
          input_size_L = input_size_L + nWordEmbed
        else
          x_f = nn.Identity()(tdnn_output_f)
          x_b = nn.Identity()(tdnn_output_b)
        end
      else -- word_vecs only
        x_f = word_vec_layer(inputs[1])
        x_b = word_vec_layer(inputs[2])
        -- x = nn.JoinTable(2)({x_f,x_b})
        input_size_L = nWordEmbed
      end
      print('Word vector size = '..input_size_L)
      if nHWLayer > 0 then
        local highway_mlp_f = HighwayMLP.mlp(input_size_L,nHWLayer)
        local highway_mlp_b = HighwayMLP.mlp(input_size_L,nHWLayer)
        highway_mlp_f.name = 'highway_f'  highway_mlp_b.name = 'highway_b'
        x_f = highway_mlp_f(x_f)
        x_b = highway_mlp_b(x_b)
      end
    else
      -- !!! will not work for multi-layer bi-directional!!!
      x_f = outputs[(L-1)*4+2*opt.use_words+2*opt.use_chars-1]
      x_b = outputs[(L-1)*4+2*opt.use_words+2*opt.use_chars]

      if dropout>0  then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end

    -- c,h from previos timesteps
    local prev_c_f = inputs[L*4+2*opt.use_words+2*opt.use_chars-3]
    local prev_c_b = inputs[L*4+2*opt.use_words+2*opt.use_chars-2]
    local prev_h_f = inputs[L*4+2*opt.use_words+2*opt.use_chars-1]
    local prev_h_b = inputs[L*4+2*opt.use_words+2*opt.use_chars]

    -- evaluate the input sums at once for efficiency
    if L ==1 then
      local i2h_f = nn.Linear(input_size_L, 4 * rnn_size)(x_f):annotate{name='i2h_f'..L} -- input to forawrd
      local h2h_f = nn.Linear(rnn_size, 4 * rnn_size)(prev_h_f):annotate{name='h2h_f'..L} -- hidden to hidden, forward
      all_input_sums_f = nn.CAddTable()({i2h_f, h2h_f}) -- input sum for forward layer

      local i2h_b = nn.Linear(input_size_L, 4 * rnn_size)(x_b):annotate{name='i2h_b'..L} -- input to backward
      local h2h_b = nn.Linear(rnn_size, 4 * rnn_size)(prev_h_b):annotate{name='h2h_b'..L} -- hidden to hidden, forward
      all_input_sums_b = nn.CAddTable()({i2h_b, h2h_b}) -- input sum for backward layer
    else  -- each recurrent layer should receive input from both forward/backward layer
      local ftof = nn.Linear(input_size_L, 4 * rnn_size)(x_f) -- prev-forward to cur-forawrd
      local btof = nn.Linear(input_size_L, 4 * rnn_size)(x_b) -- prev-backward to cur-forawrd
      local i2h_f = nn.CAddTable()({ftof, btof}):annotate{name='i2h_f'..L}
      local h2h_f = nn.Linear(rnn_size, 4 * rnn_size)(prev_h_f):annotate{name='h2h_f'..L} -- hidden to hidden, forward
      all_input_sums_f = nn.CAddTable()({i2h_f, h2h_f}) -- input sum for forward layer

      local ftob = nn.Linear(input_size_L, 4 * rnn_size)(x_f) -- prev-forward to cur-backawrd
      local btob = nn.Linear(input_size_L, 4 * rnn_size)(x_b) -- prev-backward to cur-backawrd
      local i2h_b = nn.CAddTable()({ftob, btob}):annotate{name='i2h_b'..L}
      local h2h_b = nn.Linear(rnn_size, 4 * rnn_size)(prev_h_b):annotate{name='h2h_b'..L} -- hidden to hidden, backward
      all_input_sums_b = nn.CAddTable()({i2h_b, h2h_b}) -- input sum for backward layer
    end

    ------------------------------------------- FORWARD LAYER hidden, gate, output ------------------------------------------------
    local reshaped_f = nn.Reshape(4, rnn_size)(all_input_sums_f)
    local n1_f, n2_f, n3_f, n4_f = nn.SplitTable(2)(reshaped_f):split(4)

    -- decode the gates
    local in_gate_f = nn.Sigmoid()(n1_f)
    local forget_gate_f = nn.Sigmoid()(n2_f)
    local out_gate_f = nn.Sigmoid()(n3_f)

    -- decode the write inputs
    local in_transform_f = nn.Tanh()(n4_f)

    -- perform the LSTM update
    local next_c_f           = nn.CAddTable()({
      nn.CMulTable()({forget_gate_f, prev_c_f}),
      nn.CMulTable()({in_gate_f,in_transform_f})
    })

    -- gated cells form the output
    local next_h_f = nn.CMulTable()({out_gate_f, nn.Tanh()(next_c_f)})

    table.insert(outputs, next_c_f)
    table.insert(outputs, next_h_f)

    ------------------------------------------- BACKWARD LAYER hidden, gate, output-------------------------------------------------
    local reshaped_b = nn.Reshape(4, rnn_size)(all_input_sums_b)
    local n1_b, n2_b, n3_b, n4_b = nn.SplitTable(2)(reshaped_b):split(4)

    -- decode the gates
    local in_gate_b = nn.Sigmoid()(n1_b)
    local forget_gate_b = nn.Sigmoid()(n2_b)
    local out_gate_b = nn.Sigmoid()(n3_b)

    -- decode the write inputs
    local in_transform_b = nn.Tanh()(n4_b)

    -- perform the LSTM update
    local next_c_b           = nn.CAddTable()({
      nn.CMulTable()({forget_gate_b, prev_c_b}),
      nn.CMulTable()({in_gate_b,in_transform_b})
    })

    -- gated cells form the output
    local next_h_b = nn.CMulTable()({out_gate_b, nn.Tanh()(next_c_b)})

    table.insert(outputs, next_c_b)
    table.insert(outputs, next_h_b)

    -- Omit this layer for proto.rnn
    if(L == nRNNLayer) then
      local input_for_fc_f = nn.Identity()(next_h_f)
      local input_for_fc_b = nn.Identity()(next_h_b)

      table.insert(outputs, input_for_fc_f)  -- to be used for fully connected layer input
      table.insert(outputs, input_for_fc_b)  -- to be used for fully connected layer input
    end

  end

  final_rnn = nn.gModule(inputs,outputs)

  -- graph.dot(final_rnn.fg , 'TDNNHWRNN_forward','png')


  return final_rnn

end

return biLSTMTDNN

