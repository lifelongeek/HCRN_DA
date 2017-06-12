
local uniLSTMTDNN = {}

local ok, cunn = pcall(require, 'fbcunn')
if not ok then
  LookupTable = nn.LookupTable
else
  LookupTable = nn.LookupTableGPU
end


function uniLSTMTDNN.lstmtdnn(rnn_size,nRNNLayer, dropout, nWordVocab, nWordEmbed,nCharVocab,nCharEmbed,
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

  -- there will be 2*nLayer+1 inputs
  local char_vec_layer, word_vec_layer, x, input_size_L, word_vec, char_vec
  local nHWLayer = nHWLayer or 0
  local inputs = {}    -- sets of activation is saved in here


  -- C2W
  if opt.use_chars == 1 then
    table.insert(inputs, nn.Identity()())  -- input character  (batch_sz x max_word_length) --> for 1st trial, we will provide code with batch_sz = 1
    char_vec_layer = LookupTable(nCharVocab, nCharEmbed)
    char_vec_layer.name = 'char_vecs'
  end
  if opt.use_words == 1 then
    table.insert(inputs, nn.Identity()()) -- input word (batch_sz x 1)  --> for 1st trial, we will provide code with batch_sz = 1
    word_vec_layer = LookupTable(nWordVocab, nWordEmbed)
    word_vec_layer.name = 'word_vecs'
  end

  for L = 1,nRNNLayer do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local outputs = {}
  for L = 1,nRNNLayer do
    -- the input to this layer
    if L == 1 then
      if opt.use_chars == 1 then
        char_vec = char_vec_layer(inputs[1])
        local char_tdnn = TDNN.tdnn(max_length, nCharEmbed, feature_maps, kernels)
        char_tdnn.name = 'tdnn'

        local tdnn_output = char_tdnn(char_vec)
        input_size_L = torch.Tensor(feature_maps):sum()
        if opt.use_words == 1 then
          word_vec = word_vec_layer(inputs[2])
          x = nn.JoinTable(2)({tdnn_output, word_vec})
          input_size_L = input_size_L + nWordEmbed
        else
          x = nn.Identity()(tdnn_output)
        end
      else -- word_vecs only
        x = word_vec_layer(inputs[1])
        input_size_L = nWordEmbed
      end
      print('Word vector size = '..input_size_L)
      if nHWLayer > 0 then
        local highway_mlp = HighwayMLP.mlp(input_size_L,nHWLayer)
        highway_mlp.name = 'highway'
        x = highway_mlp(x)
      end
    else
      x = outputs[(L-1)*2]
      if dropout>0  then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    
    -- c,h from previos timesteps
    local prev_c = inputs[L*2+opt.use_words+opt.use_chars-1]
    local prev_h = inputs[L*2+opt.use_words+opt.use_chars]

    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_'..L}
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_'..L}
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
    -- local reshaped = nn.Reshape(rnn_size, 4)(all_input_sums)


    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)

    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)

    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)

    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
    })

    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    table.insert(outputs, next_c)
    table.insert(outputs, next_h)

    if(L == nRNNLayer) then
      local input_for_fc = nn.Identity()(next_h)
      table.insert(outputs, input_for_fc)  -- to be used for fully connected layer input
    end
  end

  final_rnn = nn.gModule(inputs,outputs)

  return final_rnn

end

return uniLSTMTDNN

