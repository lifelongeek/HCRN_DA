-- Data loader files

local load_data = {}
load_data.__index = load_data   -- why do we need to set __index metatable as 'self'?

function load_data.create()
  -- split_fractions is e.g. {0.9, 0.05, 0.05}
  -- currently batch_sz = 1

  local self = {}
  setmetatable(self, load_data)  -- "metatable" enables operator overlading on table !
  -- See : https://gist.github.com/wikibook/5881873 for help

  -- self.padding = padding or 0
  self.padding = 0 -- for now, do not use padding (because batch_sz = 1)

  local data_dir = 'data/swda'
  ------------------------------------- Load all sentences ------------------------------------------

  -- Vectorized data version (upload all data to gpu at once !)

  -- Toyset
  --[[
  -- Whole set
  print('------ LOAD dataset for compositional word model + RNN sentence encoder ------')
  data1D_tr = torch.load(data_dir..'/data1D_tr_alltoken_alldict') onoff_utt_tr = torch.load(data_dir..'/onoff_tr_alltoken_alldict') label_tr = torch.load(data_dir..'/label_tr_alltoken_alldict')
  data1D_te = torch.load(data_dir..'/data1D_te_alltoken_alldict') onoff_utt_te = torch.load(data_dir..'/onoff_te_alltoken_alldict') label_te = torch.load(data_dir..'/label_te_alltoken_alldict')
  data1D_val = torch.load(data_dir..'/data1D_val_alltoken_alldict') onoff_utt_val = torch.load(data_dir..'/onoff_val_alltoken_alldict') label_val = torch.load(data_dir..'/label_val_alltoken_alldict')
--]]

  data1D_tr = torch.load(data_dir..'/data1D_w_tr_FINAL_PROCESS') onoff_utt_tr = torch.load(data_dir..'/onoff_utt_tr_FINAL_PROCESS') label_tr = torch.load(data_dir..'/label_tr_FINAL_PROCESS')
  data1D_te = torch.load(data_dir..'/data1D_w_te_FINAL_PROCESS') onoff_utt_te = torch.load(data_dir..'/onoff_utt_te_FINAL_PROCESS') label_te = torch.load(data_dir..'/label_te_FINAL_PROCESS')
  data1D_val = torch.load(data_dir..'/data1D_w_val_FINAL_PROCESS') onoff_utt_val = torch.load(data_dir..'/onoff_utt_val_FINAL_PROCESS') label_val = torch.load(data_dir..'/label_val_FINAL_PROCESS')
  
  -- C2W
  --[[
  dataC2W_tr = torch.load(data_dir..'/dataC2W_tr_BOWEOW')
  dataC2W_te = torch.load(data_dir..'/dataC2W_te_BOWEOW')
  dataC2W_val = torch.load(data_dir..'/dataC2W_val_BOWEOW')
  --]]


  onoff_diag_tr = torch.load(data_dir..'/onoff_diag_tr')
  onoff_diag_te = torch.load(data_dir..'/onoff_diag_te')
  onoff_diag_val = torch.load(data_dir..'/onoff_diag_val')

  -- Speakers info
  spk_tr = torch.load(data_dir..'/spk_tr')
  spk_te = torch.load(data_dir..'/spk_te')
  spk_val = torch.load(data_dir..'/spk_val')


  -- self.batches is a table of tensors
  local nTrain_utt = label_tr:size(1)  local nTest_utt = label_te:size(1)  local nValid_utt = label_val:size(1)
  local nTrain_diag = onoff_diag_tr:size(1)  local nTest_diag = onoff_diag_te:size(1)  local nValid_diag = onoff_diag_val:size(1)

  -- divide data to train/val and allocate rest to test
  self.ntrain_diag = nTrain_diag  self.ntrain_utt = nTrain_utt
  self.ntest_diag = nTest_diag  self.ntest_utt = nTest_utt
  self.nvalid_diag = nValid_diag  self.nvalid_utt = nValid_utt

  print('#Diag : (Train,Valid,Test) = '..'('..nTrain_diag..','..nValid_diag..','..nTest_diag..')')
  print('#Utterances : (Train,Valid,Test) = '..'('..nTrain_utt..','..nValid_utt..','..nTest_utt..')')

  -- dataC2W_tr:add(1)   dataC2W_te:add(1)   dataC2W_val:add(1)
  if(opt.gpuid>=0) then
    data1D_tr = data1D_tr:float():cuda()  label_tr = label_tr:float():cuda()  onoff_diag_tr = onoff_diag_tr:float():cuda()   spk_tr = spk_tr:float():cuda() -- dataC2W_tr = dataC2W_tr:float():cuda()
    data1D_te = data1D_te:float():cuda()  label_te = label_te:float():cuda()  onoff_diag_te = onoff_diag_te:float():cuda()   spk_te = spk_te:float():cuda()-- dataC2W_te = dataC2W_te:float():cuda()
    data1D_val = data1D_val:float():cuda()  label_val = label_val:float():cuda()  onoff_diag_val = onoff_diag_val:float():cuda()   spk_val = spk_val:float():cuda()-- dataC2W_val = dataC2W_val:float():cuda()
  end
  self.data1D_tr = data1D_tr  self.onoff_utt_tr = onoff_utt_tr  self.label_tr = label_tr  self.onoff_diag_tr = onoff_diag_tr  -- self.dataC2W_tr = dataC2W_tr
  self.data1D_te = data1D_te  self.onoff_utt_te = onoff_utt_te  self.label_te = label_te  self.onoff_diag_te = onoff_diag_te  -- self.dataC2W_te = dataC2W_te
  self.data1D_val = data1D_val  self.onoff_utt_val = onoff_utt_val  self.label_val = label_val  self.onoff_diag_val = onoff_diag_val  -- self.dataC2W_val = dataC2W_val

  self.split_sizes_diag = {self.ntrain_diag, self.ntest_diag, self.nvalid_diag}
  self.split_sizes_utt = {self.ntrain_utt, self.ntest_utt, self.nvalid_utt}
  self.batch_ix_diag = {0,0,0}
  -- self.batch_ix_utt = {0,0,0}  -- not used

  -- create random permutation index (will be permuted again for every epoch)
  self.randIdx_diag = torch.randperm(self.ntrain_diag):long() --indexing requires longtensor()
  -- do not randomly permute utterance order

  print('data load done.')
  collectgarbage()
  return self
end


function load_data:next_dialogue(split_index)
  -- split_index is integer: 1 = train, 2 = test
  self.batch_ix_diag[split_index]  = self.batch_ix_diag[split_index]  + 1
  if self.batch_ix_diag[split_index]  > self.split_sizes_diag[split_index] then
    self.batch_ix_diag[split_index] = 1
    self.randIdx_diag = torch.randperm(self.ntrain_diag) -- new random permutated index
  end
  local ix = self.batch_ix_diag[split_index]

  -- Note : for given 2D matrix A, B = A[{{x,y}}] selects from row x to row y of A.


  if(split_index == 1) then
    return self.data1D_tr[{{self.onoff_utt_tr[self.onoff_diag_tr[self.randIdx_diag[ix]][1]][1] , self.onoff_utt_tr[self.onoff_diag_tr[self.randIdx_diag[ix]][2]][2]}}],
      self.label_tr[{{self.onoff_diag_tr[self.randIdx_diag[ix]][1] , self.onoff_diag_tr[self.randIdx_diag[ix]][2]}}],   -- Same for all code
      spk_tr[{{self.onoff_diag_tr[self.randIdx_diag[ix]][1] , self.onoff_diag_tr[self.randIdx_diag[ix]][2]}}]
  elseif(split_index == 2) then
    return self.data1D_te[{{self.onoff_utt_te[self.onoff_diag_te[ix][1]][1] , self.onoff_utt_te[self.onoff_diag_te[ix][2]][2]}}],
      self.label_te[{{self.onoff_diag_te[ix][1] , self.onoff_diag_te[ix][2]}}],  -- Same for all code
      spk_te[{{self.onoff_diag_te[ix][1] , self.onoff_diag_te[ix][2]}}]
  elseif(split_index == 3) then
    return self.data1D_val[{{self.onoff_utt_val[self.onoff_diag_val[ix][1]][1] , self.onoff_utt_val[self.onoff_diag_val[ix][2]][2]}}],
      self.label_val[{{self.onoff_diag_val[ix][1] , self.onoff_diag_val[ix][2]}}],  -- Same for all code
      spk_val[{{self.onoff_diag_val[ix][1] , self.onoff_diag_val[ix][2]}}]
  end
end

return load_data


