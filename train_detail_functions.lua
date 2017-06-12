-- require 'util.Embedding'
require 'util.misc'

require 'nn'
require 'nngraph'
nngraph.setDebug(true)  -- for debug

GRU_uni_multi_diagRNN = require 'model.GRU_uni_multi_diagRNN'

local model_utils = require 'util.model_utils'


function get_network()
  init_act_uRNN = {}
  init_act_global_uRNN = {}
  init_act_dRNN = {}
  init_act_global_dRNN = {}


  print('Making compositional word model & sentence encoder')
  -- Step1) Make neural activation variables

  -- for utterance RNN
  for layeridx=1,opt.L_uRNN do
    local h_init_uRNN=  torch.zeros(1, opt.H_uRNN)
    if opt.gpuid >=0 then h_init_uRNN = h_init_uRNN:cuda() end
    table.insert(init_act_uRNN,h_init_uRNN:clone()) -- block activation_forward
  end
  init_act_global_uRNN = clone_list(init_act_uRNN)

  -- for dialogue RNN
  for layeridx=1,opt.L_dRNN do
    local h_init_dRNN=  torch.zeros(1, opt.H_dRNN)
    if opt.gpuid >=0 then h_init_dRNN = h_init_dRNN:cuda() end
    table.insert(init_act_dRNN,h_init_dRNN:clone()) -- block activation_forward
  end
  init_act_global_dRNN = clone_list(init_act_dRNN)


  -- initialization
  if opt.startEpoch == 1 then
    -- ver1 Initial
    protos_diag = {}
    protos_diag.rnn = GRU_uni_multi_diagRNN.gru(nUttEmbed, opt.L_dRNN, opt.H_dRNN, opt.dropout, opt.nOutput)
    protos_diag.rnn.name = 'diagRNN'
    protos_diag.criterion = nn.ClassNLLCriterion()

    -- Step2) Build compositional word model + sentence encoder
    wordVec = torch.load('exp_sent_train/'..opt.sent_train_expnum..'/wordVec')  -- prestored word vector (dimension = nVocabWord x wordvec_sz (=H_wRNN)
    print('successfully load word vector from sentence-level training')

    protos_utt = torch.load('exp_sent_train/'..opt.sent_train_expnum..'/protos_utt_mv')

    loss_tr_epoch = torch.zeros(opt.maxEpoch)  loss_te_epoch = torch.zeros(opt.maxEpoch)  loss_val_epoch = torch.zeros(opt.maxEpoch)
    acc_tr_epoch = torch.zeros(opt.maxEpoch)  acc_te_epoch = torch.zeros(opt.maxEpoch)   acc_val_epoch = torch.zeros(opt.maxEpoch)
  else
    -- params_tdnnhwrnn_load = torch.load(saveDir..'/params_tdnnhwrnn')  params_cRNN_load = torch.load(saveDir..'/params_cRNN')
    protos_diag = torch.load(saveDir..'/protos_diag')   protos_utt= torch.load(saveDir..'/protos_utt')  wordVec = torch.load('exp_sent_train/'..opt.sent_train_expnum..'/wordVec')
    loss_tr_epoch = torch.load(saveDir..'/loss_tr_epoch')     loss_te_epoch = torch.load(saveDir..'/loss_te_epoch')  loss_val_epoch = torch.load(saveDir..'/loss_val_epoch')
    acc_tr_epoch = torch.load(saveDir..'/acc_tr_epoch')      acc_te_epoch = torch.load(saveDir..'/acc_te_epoch')  acc_val_epoch = torch.load(saveDir..'/acc_val_epoch')
    collectgarbage()
  end

  if opt.gpuid >= 0 then
    for k,v in pairs(protos_utt) do v:cuda() end
    for k,v in pairs(protos_diag) do v:cuda() end
  end

  -- flatten all parameters
  params_uRNN, grad_uRNN = model_utils.combine_all_parameters(protos_utt.rnn)
  params_dRNN, grad_dRNN = model_utils.combine_all_parameters(protos_diag.rnn)
  np_uRNN = params_uRNN:nElement()
  np_dRNN = params_dRNN:nElement()

  print('loaded compositional word model with #layer =' .. opt.L_uRNN .. ', #parameters = ' .. np_uRNN)
  print('created an dialogueRNN with ' .. opt.L_dRNN .. ' layers, #parameters = ' .. np_dRNN)


  if opt.startEpoch == 1 then params_dRNN:uniform(-opt.dRNN_stddev,opt.dRNN_stddev)  end
  collectgarbage()

  -- global parameters
  grad_dRNN_sum = torch.Tensor(np_dRNN)
  grad_uRNN_sum = torch.Tensor(np_uRNN)  grad_uRNN_sum_sum = torch.Tensor(np_uRNN)

  -- activation storage
  wordvec_sz = wordVec:size(2)
  print('Word vector size = '..wordvec_sz)

  wordVec_store = torch.Tensor(opt.maxWordperSent, wordvec_sz):zero()
  uRNN_act = torch.Tensor(opt.maxWordperSent*opt.L_uRNN, opt.H_uRNN):zero()

  if opt.gpuid>=0 then
    wordVec_store = wordVec_store:float():cuda()
    uRNN_act = uRNN_act:float():cuda()
    grad_dRNN_sum = grad_dRNN_sum:float():cuda()
    grad_uRNN_sum = grad_uRNN_sum:float():cuda()  grad_uRNN_sum_sum = grad_uRNN_sum_sum:float():cuda()
  end

  zero_dummy = torch.Tensor(1,opt.H_uRNN):zero()
  spk_vec = torch.Tensor(3,2):zero()
  spk_vec[2][1] = 1 spk_vec[3][2] = 1
  if(opt.gpuid>=0) then
    zero_dummy = zero_dummy:float():cuda()
    spk_vec = spk_vec:float():cuda()
  end

  -- Step4) clone as word RNN && utterance RNN & dialogue RNN
  clones_utt = {}
  for name,proto in pairs(protos_utt) do   --protos.name(1) = rnn, protos.name(2) = criterion
    print('cloning word-' .. name .. ' as much as '..opt.maxWordperSent..'times(This may take a time)')
    clones_utt[name] = model_utils.clone_many_times(proto, opt.maxWordperSent, not proto.parameters) -- parameters are shared
  end
  print('end of cloning')

  clones_diag = {}
  for name,proto in pairs(protos_diag) do   --protos.name(1) = rnn, protos.name(2) = criterion
    print('cloning utterance-' .. name .. ' as much as '..opt.maxUttperDiag..'times(This may take a time)')
    clones_diag[name] = model_utils.clone_many_times(proto, opt.maxUttperDiag, not proto.parameters) -- parameters are shared
  end
  print('end of cloning')

  -- Training monitoring
  confusion_tr = optim.ConfusionMatrix(opt.nOutput)
  confusion_te = optim.ConfusionMatrix(opt.nOutput)
  confusion_val = optim.ConfusionMatrix(opt.nOutput)
  acc_tr = 0

  -- Note : global variable definition is moved to upper part
end


function feval_dRNN(x)
  if x ~= params_dRNN then params_dRNN:copy(x) end

  local batch_sz_real = math.min(loader.ntrain_diag - loader.batch_ix_diag[1], opt.batch_sz)
  if(batch_sz_real == 0) then batch_sz_real = opt.batch_sz end
  loss_tr_local = 0

  -- dialogue RNN
  grad_dRNN_sum:zero()
  if(epoch >= opt.dRNN_only_epoch) then
    -- utterance RNN
    grad_uRNN_sum:zero()  grad_uRNN_sum_sum:zero()
  end

  for diagIdx = 1,batch_sz_real do
    grad_dRNN:zero()

    local predictions = {}
    local uttEmbeddings = {}

    ------------------ Get input ---------------------

    -- Ver1 : with C2W
    -- utt_tr_words_tot,targets_tr,utt_tr_C2W_tot = loader:next_dialogue(1)

    -- Ver2 : without C2W
    -- utt_tr_words_tot,targets_tr = loader:next_dialogue(1)

    -- Ver3 : with speaker info
    utt_tr_words_tot, targets_tr,spk_tr_utt = loader:next_dialogue(1)

    assert(targets_tr:size(1) == spk_tr_utt:size(1))

    nUtt = targets_tr:size(1)

    local ix = loader.batch_ix_diag[1]

    -- print('#Utterances in dialogue='..nUtt)

    ----------------- Feedforward --------------------
    rnn_act_diag = {[0] = init_act_global_dRNN}  -- dRNN activation

    rnn_act_utt_allutt = {}
    -- wordvec_store_allutt = {}

    -- for GRU Layers
    nWords_processed = 0
    utterance_start_number = loader.onoff_diag_tr[loader.randIdx_diag[ix]][1]
    for uttIdx=1,nUtt do
      uRNN_act:zero()
      nWord = loader.onoff_utt_tr[utterance_start_number+uttIdx-1][2] - loader.onoff_utt_tr[utterance_start_number+uttIdx-1][1] + 1
      -- print(nWord)
      utt_tr_word = utt_tr_words_tot[{{nWords_processed + 1, nWords_processed + nWord}}] -- 2D tensor
      -- utt_tr_C2W = utt_tr_C2W_tot[{{},{nWords_processed + 1, nWords_processed + nWord}}] -- 3D tensor

      --[[
      if(diagIdx == 1 and uttIdx == 5) then
        print('Forward, utt_tr_word @uttIdx=5 = ')
        torch.save('uRNN_tr_word_FR',utt_tr_word)
      end
      --]]

      nWords_processed = nWords_processed + nWord
      clones_diag.rnn[uttIdx]:training()  -- load is negligible

      rnn_act_utt = {[0] = init_act_global_uRNN}

      for wordIdx = 1,nWord do
        -- utterance RNN
        clones_utt.rnn[wordIdx]:training() --load is negligible

        --Ver2 : I don't like this kind of long implementation but it is for table.insert(rnn_act_utt_allutt,) work
        if(wordIdx >= 2) then
          uRNN_act_table = {}
          for i=1,opt.L_uRNN do
            uRNN_act_table[i] = uRNN_act[(wordIdx-2)*opt.L_uRNN + i]
          end
          --[[
          if(diagIdx == 1 and uttIdx == 5 and wordIdx == 2) then
            print('Forward, uRNN_act_table @diagIdx=1, uttIdx=5, wordIdx = 2')
            torch.save('uRNN_act_table_FR',uRNN_act_table)
          end
          --]]
          lst_utt = clones_utt.rnn[wordIdx]:forward{wordVec[utt_tr_word[wordIdx][1]], unpack(uRNN_act_table)}
        else
          lst_utt = clones_utt.rnn[wordIdx]:forward{wordVec[utt_tr_word[wordIdx][1]], unpack(init_act_global_uRNN)}
        end

        --uRNN activation : Ver2
        for i=1,opt.L_uRNN do
          uRNN_act[(wordIdx-1)*opt.L_uRNN + i] = lst_utt[i]:clone()
        end
      end -- end of forward word iteration
      --[[
      if(diagIdx == 1 and uttIdx == 5) then
        print('Forward, uRNN_act @uttIdx=5 = ')
        torch.save('uRNN_act_FR',uRNN_act)
      end
      --]]

      if (epoch >= opt.dRNN_only_epoch) then
        -- hwoutput_allutt[uttIdx] = hwoutput  -- store activation for using in backpropagation phase
        -- wordVec_store_allutt[uttIdx] = wordVec[utt_tr_word[wordIdx][1]]:clone()  -- store activation for using in backpropagation phase
        rnn_act_utt_allutt[uttIdx] = uRNN_act:clone()
      end

      -- making utterance vector
      uttEmbed = lst_utt[#lst_utt]
      -- Ver1
      -- uttEmbeddings[uttIdx] = uttEmbed
      -- Ver2
      uttEmbeddings[uttIdx] = uttEmbed:clone()

      --[[
      if(diagIdx == 1 and uttIdx == 5) then
        print('Forward, uttEmbed @uttIdx=5 = ')
        torch.save('uttEmbed_FR',uttEmbed)
      end
      --]]

      -- dialogue RNN forward propagation
      -- Ver1 : without speaker info
      -- lst_diag = clones_diag.rnn[uttIdx]:forward{uttEmbed,unpack(rnn_act_diag[uttIdx-1])}
      -- Ver2 : with speaker info
      if(uttIdx >1 and spk_tr_utt[uttIdx][1] == spk_tr_utt[uttIdx-1][1]) then  -- Same speaker
        spk_vec_in = spk_vec[2]
      elseif(uttIdx >1 and spk_tr_utt[uttIdx][1] ~= spk_tr_utt[uttIdx-1][1]) then  -- Different speaker
        spk_vec_in = spk_vec[3]
      elseif(uttIdx == 1) then
        spk_vec_in = spk_vec[1]
      else
        assert(0)
      end
      lst_diag = clones_diag.rnn[uttIdx]:forward{uttEmbed,spk_vec_in, unpack(rnn_act_diag[uttIdx-1])}

      rnn_act_diag[uttIdx] = {}

      for i=1,#init_act_dRNN do table.insert(rnn_act_diag[uttIdx], lst_diag[i]) end
      prediction = lst_diag[#lst_diag]

      --Ver1
      -- predictions[uttIdx] = prediction
      --Ver2
      predictions[uttIdx] = prediction:clone()

      --[[
      if(diagIdx == 1 and uttIdx == 5) then
        print('Forward, prediction @uttIdx=5 = ')
        print(prediction)
      end
      --]]

      target = targets_tr[uttIdx][1]


      loss = clones_diag.criterion[uttIdx]:forward(prediction, target)

      loss_tr_local = loss_tr_local + loss

      local correct
      _, predIdx = torch.max(prediction,1)

      predIdx = predIdx[1]
      if(predIdx == target) then correct = 1  else correct = 0 end

      acc_tr = acc_tr + correct
      confusion_tr:add(predIdx, target)

      correct_each_tr[loader.onoff_diag_tr[loader.randIdx_diag[ix]][1] + uttIdx-1] = correct
      loss_each_tr[loader.onoff_diag_tr[loader.randIdx_diag[ix]][1] + uttIdx-1] = loss
    end -- end of forward utterance iteration

    -- print('nWords_processed = '..nWords_processed..' VS '..'utt_tr_words_tot:size(1) = '..utt_tr_words_tot:size(1))
    assert(nWords_processed == utt_tr_words_tot:size(1))

    ---------------- Backpropagation ----------------
    -- LSTM layers
    local drnn_err_diag = {[nUtt] = clone_list(init_act_dRNN,true)}
    if(epoch >= opt.dRNN_only_epoch) then
      grad_uRNN_sum:zero()  -- for dialogue
    end
    grad_dRNN:zero()

    for uttIdx=nUtt,1,-1 do
      grad_uRNN:zero()  -- do this regardless of opt.dRNN_only or not (prevent garbage computation)

      uttEmbed = uttEmbeddings[uttIdx]
      prediction = predictions[uttIdx]

      --[[
      if(diagIdx == 1 and uttIdx == 5) then
        print('Backward, uttEmbed @uttIdx=5 = ')
        torch.save('uttEmbed_BW',uttEmbed)

        print('Backward, prediction @uttIdx=5 = ')
        print(prediction)
      end
      --]]


      local doutput_uttIdx = clones_diag.criterion[uttIdx]:backward(prediction, targets_tr[uttIdx][1])
      table.insert(drnn_err_diag[uttIdx], doutput_uttIdx)

      -- Ver1 : Without speaker info
      -- dlst_diag = clones_diag.rnn[uttIdx]:backward({uttEmbed, unpack(rnn_act_diag[uttIdx-1])}, drnn_err_diag[uttIdx]) -- EBP of diagRNN
      -- Ver2 : with speaker info
      if(uttIdx >1 and spk_tr_utt[uttIdx][1] == spk_tr_utt[uttIdx-1][1]) then  -- Same speaker
        spk_vec_in = spk_vec[2]
      elseif(uttIdx >1 and spk_tr_utt[uttIdx][1] ~= spk_tr_utt[uttIdx-1][1]) then  -- Different speaker
        spk_vec_in = spk_vec[3]
      elseif(uttIdx == 1) then
        spk_vec_in = spk_vec[1]
      else
        assert(0)
      end
      dlst_diag = clones_diag.rnn[uttIdx]:backward({uttEmbed, spk_vec_in, unpack(rnn_act_diag[uttIdx-1])}, drnn_err_diag[uttIdx]) -- EBP of diagRNN

      drnn_err_diag[uttIdx-1] = {}
      for k,v in pairs(dlst_diag) do
        --Ver1 : without speaker info to diagRNN
        --[[
        if k > 1 then -- gradient at layer 1 will be explicitly back-propagated by other variable
          drnn_err_diag[uttIdx-1][k-1] = v
        end
        --]]  
        --Ver2 : with speaker info to diagRNN
        if k > 2 then -- gradient at layer 1 will be explicitly back-propagated by other variable
          drnn_err_diag[uttIdx-1][k-2] = v
        end
      end

      if(epoch >=opt.dRNN_only_epoch) then
        uRNN_act = rnn_act_utt_allutt[uttIdx]  -- stored activation from forward propagation
        --[[
        if(diagIdx == 1 and uttIdx == 5) then
          print('Backward, uRNN_act @uttIdx=5 = ')
          torch.save('uRNN_act_BW',uRNN_act)
        end
        --]]

        nWord = loader.onoff_utt_tr[utterance_start_number+uttIdx-1][2] -loader.onoff_utt_tr[utterance_start_number+uttIdx-1][1] + 1

        -- Get utterance as reverse order (for BPTT)
        utt_tr_word = utt_tr_words_tot[{{nWords_processed -nWord +1, nWords_processed}}] -- 2D tensor
        -- utt_tr_C2W = utt_tr_C2W_tot[{{},{nWords_processed -nWord + 1, nWords_processed}}] -- 3D tensor

        --[[
        if(diagIdx == 1 and uttIdx == 5) then
          print('Backward, utt_tr_word @uttIdx=5 = ')
          torch.save('utt_tr_word_BW',utt_tr_word)
        end
        --]]

        nWords_processed = nWords_processed - nWord

        local ddiag_dutt = dlst_diag[1]  --con : context RNN, com : combiner

        local drnn_err_utt = {[nWord] = clone_list(init_act_uRNN)}  -- initial error of BPTT : zeros()

        -- utterance RNN + TDNNHW
        for wordIdx = nWord,1,-1 do
          if(wordIdx == nWord) then
            table.insert(drnn_err_utt[wordIdx], ddiag_dutt)
          else
            table.insert(drnn_err_utt[wordIdx], zero_dummy)
          end

          -- utterance RNN
          if(wordIdx >= 2) then
            uRNN_act_table = {}
            for i=1,opt.L_uRNN do
              uRNN_act_table[i] = uRNN_act[(wordIdx-2)*opt.L_uRNN + i]
            end
            --[[
            if(diagIdx == 1 and uttIdx == 5 and wordIdx == 2) then
              print('Backward, uRNN_act_table @diagIdx=1, uttIdx=5, wordIdx = 2')
              torch.save('uRNN_act_table_BW',uRNN_act_table)
            end
            --]]
            dlst_utt = clones_utt.rnn[wordIdx]:backward({wordVec[utt_tr_word[wordIdx][1]], unpack(uRNN_act_table)}, drnn_err_utt[wordIdx])
          else
            dlst_utt = clones_utt.rnn[wordIdx]:backward({wordVec[utt_tr_word[wordIdx][1]], unpack(init_act_global_uRNN)}, drnn_err_utt[wordIdx])
          end

          drnn_err_utt[wordIdx-1] = {}
          th_num = 1
          for k,v in pairs(dlst_utt) do
            if k> th_num then
              drnn_err_utt[wordIdx-1][k-th_num] = v
            end
          end
        end -- end of backward word iteration

        if(epoch >= opt.dRNN_only_epoch) then
          grad_uRNN_sum:add(grad_uRNN:mul(1/nWord))
        end

      end -- end of if(epoch >= opt.dRNN_only_epoch)
    end -- end of utterance iteration
    grad_dRNN_sum:add(grad_dRNN:mul(1/nUtt))
    if(epoch >= opt.dRNN_only_epoch) then
      grad_uRNN_sum_sum:add(grad_uRNN_sum:mul(1/nUtt))
    end

    nUtt_processed_tr = nUtt_processed_tr + nUtt -- global variable (used in main.lua)
  end  --end of dialogue iteration

  -- divide by minibatch size
  -- dialogue RNN
  grad_dRNN_sum:div(batch_sz_real)  -- batch_sz_real = #dialogues to average
  grad_dRNN:copy(grad_dRNN_sum)
  grad_dRNN:clamp(-opt.grad_clip, opt.grad_clip)

  if(epoch >= opt.dRNN_only_epoch) then
    -- utterance RNN
    grad_uRNN_sum_sum:div(batch_sz_real)  grad_uRNN:copy(grad_uRNN_sum_sum)
    grad_uRNN:clamp(-opt.grad_clip, opt.grad_clip)
  end

  return loss_tr_local, grad_dRNN
end


function feval_uRNN(x)
  if x ~= params_uRNN then params_uRNN:copy(x) end

  return 0, grad_uRNN
end



-- First finish test of training, and then write code for test/valid
function test()
  local loss_te_local = 0
  local acc_te = 0

  for diagIdx = 1,loader.ntest_diag do
    ------------------ Get input ---------------------

    -- utt_te_words_tot,targets_te,utt_te_C2W_tot = loader:next_dialogue(2)
    -- utt_te_words_tot,targets_te = loader:next_dialogue(2)
    utt_te_words_tot, targets_te,spk_te_utt = loader:next_dialogue(2)

    nUtt = targets_te:size(1)

    local ix = loader.batch_ix_diag[2]

    -- print('#Utterances in dialogue='..nUtt)

    ----------------- Feedforward --------------------
    rnn_act_diag = {[0] = init_act_global_dRNN}

    -- for GRU Layers
    nWords_processed = 0
    utterance_start_number = loader.onoff_diag_te[ix][1] -- ix : dialogue index

    for uttIdx=1,nUtt do
      nWord = loader.onoff_utt_te[utterance_start_number+uttIdx-1][2] - loader.onoff_utt_te[utterance_start_number+uttIdx-1][1] + 1

      -- print(nWord)

      utt_te_word = utt_te_words_tot[{{nWords_processed + 1, nWords_processed + nWord}}] -- 2D tensor
      -- utt_te_C2W = utt_te_C2W_tot[{{},{nWords_processed + 1, nWords_processed + nWord}}] -- 3D tensor
      nWords_processed = nWords_processed + nWord
      clones_diag.rnn[uttIdx]:evaluate()  -- load is negligible

      rnn_act_utt = {[0] = init_act_global_uRNN}

      -- initialize utterance vector
      for wordIdx = 1,nWord do
        --print('nWord = '..nWord)

        -- utterance RNN
        clones_utt.rnn[wordIdx]:evaluate() --load is negligible

        lst_utt = clones_utt.rnn[wordIdx]:forward{wordVec[utt_te_word[wordIdx][1]], unpack(rnn_act_utt[wordIdx-1])}

        rnn_act_utt[wordIdx] = {}
        for i=1,#init_act_uRNN do
          table.insert(rnn_act_utt[wordIdx],lst_utt[i])
        end
      end

      -- making utterance vector
      uttEmbed = lst_utt[#lst_utt]

      -- Ver1 : without speaker info
      -- lst_diag = clones_diag.rnn[uttIdx]:forward{uttEmbed,unpack(rnn_act_diag[uttIdx-1])}
      -- Ver2 : with speaker info
      if(uttIdx >1 and spk_te_utt[uttIdx][1] == spk_te_utt[uttIdx-1][1]) then  -- Same speaker
        spk_vec_in = spk_vec[2]
      elseif(uttIdx >1 and spk_te_utt[uttIdx][1] ~= spk_te_utt[uttIdx-1][1]) then  -- Different speaker
        spk_vec_in = spk_vec[3]
      elseif(uttIdx == 1) then
        spk_vec_in = spk_vec[1]
      else
        assert(0)
      end
      lst_diag = clones_diag.rnn[uttIdx]:forward{uttEmbed,spk_vec_in, unpack(rnn_act_diag[uttIdx-1])}

      rnn_act_diag[uttIdx] = {}

      for i=1,#init_act_dRNN do table.insert(rnn_act_diag[uttIdx], lst_diag[i]) end
      prediction = lst_diag[#lst_diag]
      target = targets_te[uttIdx][1]

      loss = clones_diag.criterion[uttIdx]:forward(prediction, target)
      loss_te_local = loss_te_local + loss

      local correct
      _, predIdx = torch.max(prediction,1)

      predIdx = predIdx[1]
      if(predIdx == target) then correct = 1  else correct = 0 end

      acc_te = acc_te + correct

      confusion_te:add(predIdx, target)

      correct_each_te[loader.onoff_diag_te[ix][1] + uttIdx-1] = correct
      loss_each_tr[loader.onoff_diag_te[ix][1] + uttIdx-1] = loss
    end -- end of utterance iteration

    assert(nWords_processed == utt_te_words_tot:size(1))
  end -- end of dialogue iteration
  loss_te_local = loss_te_local/loader.ntest_utt
  acc_te = acc_te/loader.ntest_utt*100

  return loss_te_local, acc_te
end


function valid()
  local loss_val_local = 0
  local acc_val = 0

  for diagIdx = 1,loader.nvalid_diag do
    ------------------ Get input ---------------------

    -- utt_val_words_tot,targets_val,utt_val_C2W_tot = loader:next_dialogue(3)
    -- utt_val_words_tot,targets_val = loader:next_dialogue(3)
    utt_val_words_tot, targets_val,spk_val_utt = loader:next_dialogue(3)

    nUtt = targets_val:size(1)

    local ix = loader.batch_ix_diag[3]

    -- print('#Utterances in dialogue='..nUtt)

    ----------------- Feedforward --------------------
    rnn_act_diag = {[0] = init_act_global_dRNN}

    -- for GRU Layers
    nWords_processed = 0
    utterance_start_number = loader.onoff_diag_val[ix][1] -- ix : dialogue index
    for uttIdx=1,nUtt do
      nWord = loader.onoff_utt_val[utterance_start_number+uttIdx-1][2] - loader.onoff_utt_val[utterance_start_number+uttIdx-1][1] + 1
      -- print(nWord)
      utt_val_word = utt_val_words_tot[{{nWords_processed + 1, nWords_processed + nWord}}] -- 2D tensor
      -- utt_val_C2W = utt_val_C2W_tot[{{},{nWords_processed + 1, nWords_processed + nWord}}] -- 3D tensor
      nWords_processed = nWords_processed + nWord
      clones_diag.rnn[uttIdx]:evaluate()  -- load is negligible

      rnn_act_utt = {[0] = init_act_global_uRNN}

      -- initialize utterance vector
      for wordIdx = 1,nWord do
        --print('nWord = '..nWord)
        -- utterance RNN
        clones_utt.rnn[wordIdx]:evaluate() --load is negligible
        lst_utt = clones_utt.rnn[wordIdx]:forward{wordVec[utt_val_word[wordIdx][1]], unpack(rnn_act_utt[wordIdx-1])}

        rnn_act_utt[wordIdx] = {}
        for i=1,#init_act_uRNN do
          table.insert(rnn_act_utt[wordIdx],lst_utt[i])
        end
      end

      -- making utterance vector
      uttEmbed = lst_utt[#lst_utt]

      -- Ver1 : without speaker info
      -- lst_diag = clones_diag.rnn[uttIdx]:forward{uttEmbed,unpack(rnn_act_diag[uttIdx-1])}
      -- Ver2 : with speaker info
      if(uttIdx >1 and spk_val_utt[uttIdx][1] == spk_val_utt[uttIdx-1][1]) then  -- Same speaker
        spk_vec_in = spk_vec[2]
      elseif(uttIdx >1 and spk_val_utt[uttIdx][1] ~= spk_val_utt[uttIdx-1][1]) then  -- Different speaker
        spk_vec_in = spk_vec[3]
      elseif(uttIdx == 1) then
        spk_vec_in = spk_vec[1]
      else
        assert(0)
      end
      lst_diag = clones_diag.rnn[uttIdx]:forward{uttEmbed,spk_vec_in, unpack(rnn_act_diag[uttIdx-1])}

      rnn_act_diag[uttIdx] = {}

      for i=1,#init_act_dRNN do table.insert(rnn_act_diag[uttIdx], lst_diag[i]) end
      prediction = lst_diag[#lst_diag]
      target = targets_val[uttIdx][1]
      loss = clones_diag.criterion[uttIdx]:forward(prediction, target)
      loss_val_local = loss_val_local + loss

      local correct
      _, predIdx = torch.max(prediction,1)

      predIdx = predIdx[1]
      if(predIdx == target) then correct = 1  else correct = 0 end

      acc_val = acc_val + correct
      confusion_val:add(predIdx, target)

      correct_each_val[loader.onoff_diag_val[ix][1] + uttIdx-1] = correct
      loss_each_val[loader.onoff_diag_val[ix][1] + uttIdx-1] = loss
    end

    assert(nWords_processed == utt_val_words_tot:size(1))
  end
  loss_val_local = loss_val_local/loader.nvalid_utt
  acc_val = acc_val/loader.nvalid_utt*100

  return loss_val_local, acc_val
end
