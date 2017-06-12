require 'torch'
require 'nn'
require 'optim'
require 'nngraph'
nngraph.setDebug(true)  -- for debug

local load_data = require 'util.load_data'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a DA tracker started from pre-trained compositional char + compositional word model.')
cmd:text()
cmd:text('Options')

-- General
cmd:option('-sent_train_expnum',-1,'Sentence-level training experiment number : source for pre-trained TDNNHW + utterance RNN')

cmd:option('-expnum',1,'experiment number : make sure always assign exp#. Check with your experiment sheet')
cmd:option('-print_every',10,'how frequently print progress')
cmd:option('-batch_sz',1,'minibatch size(#dialogues/update)')
cmd:option('-maxEpoch',500,'maximum #epoch')
cmd:option('-startEpoch',1,'startepoch. if 1, starts from random initialization, else, resume previous training')
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-cudnn',1,'use cudnn (0=no)')
cmd:option('-momentum',0.9,'momentum for sgd, note that momentum will not be applied from epoch2')
cmd:option('-wdcay',0,'wdcay for sgd')
cmd:option('-grad_clip',5,'element-wise gradient clipthreshold')
cmd:option('-dRNN_only_epoch',5,'#epoch to train dRNN only (freeze sentence-level training architecture)')

--Architecture
-- for compositional character RNN
--[[
cmd:option('-H_wRNN',128, '#Hidden neurons in RNN compositional word model')
cmd:option('-L_wRNN',1, '#Layer in RNN compositional word model')
cmd:option('-maxCharperWord',21,'maximum number of char in word. rnn is cloned as much as this sequence')  -- 20 + 1(End of word token)
cmd:option('-nCharVocab',34)  -- 32-->34 (additional . character && <eow> token)
cmd:option('-nCharEmbed',15)
--]]

-- for utterance RNN
cmd:option('-H_uRNN',128,'#Hidden neurons in RNN sentence encoder')
-- cmd:option('-enc_bidirection',false,'bi-directional computation for RNN encoder')
cmd:option('-enc_meanpooling',false,'mean pooling for RNN encoder')
cmd:option('-maxWordperSent',112,'maximum number of word in sentence. rnn is cloned as much as this sequence')
cmd:option('-L_uRNN',1,'#RNN(LSTM) layers of sentence encoding RNN')

-- for dialogueRNN
cmd:option('-H_dRNN',256,'#Hidden neurons in context RNN')
cmd:option('-L_dRNN',2,'#RNN(LSTM) layers of context RNN')
cmd:option('-maxUttperDiag',412,'maximum number of word in sentence. rnn is cloned as much as this sequence')
cmd:option('-dRNN_stddev',0.1,'initialization stddev of context RNN')

-- Fully connected
cmd:option('-H_fc',128)
cmd:option('-L_fc',2)
cmd:option('-fc_stddev',0.01,'initialization stddev of fc')
cmd:option('-nOutput',42,'#DA class')


--Learning parameter
-- cmd:option('-lRate0',0.002,'initial learning rate')
-- cmd:option('-lRate0',0.01,'initial learning rate')
cmd:option('-lRate0',0.05,'initial learning rate')

cmd:option('-dropout',0,'apply dropout except recurrent connection (see paper : RNN regularization)')
cmd:option('-optim_method','adadelta','optimization method : sgd, rmsprop, (l-bfgs, adagrad, adam ...)')

--GPU/CPU
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')

cmd:text()
opt = cmd:parse(arg)

maxval = -100
torch.manualSeed(opt.seed)

print('opt.maxWordperSent = '..opt.maxWordperSent)
print('opt.maxUttperDiag = '..opt.maxUttperDiag)



if opt.gpuid >=0 then
  require 'cunn'
  require 'cutorch'
end
require 'train_detail_functions'

print('Experiment# = '..opt.expnum)
saveDir = 'exp/'..opt.expnum
os.execute('mkdir -p '..saveDir)

-- GPU setting
require 'util.gpu_setup'


print('-- This code is intended for dialogue-level training of whole systsem --')
print('-- At early phase, it will train DAtracker only with 5epoch, and then joint training of whole architecture is applied')

-- Data loading
print('-- Load dataset from torch7 file(converted from matlab) --')
loader = load_data.create()

-- Check utterance embedding size
nUttEmbed = opt.H_uRNN
print('(BUILD FROM C2W-RNNEnc)Utterance embedding size = '..nUttEmbed)

loss_each_tr = torch.zeros(loader.ntrain_utt)  loss_each_te = torch.zeros(loader.ntest_utt)  loss_each_val =  torch.zeros(loader.nvalid_utt)
correct_each_tr = torch.zeros(loader.ntrain_utt) correct_each_te = torch.zeros(loader.ntest_utt) correct_each_val = torch.zeros(loader.nvalid_utt)

-- Network loading
print('-- Building network  --')
get_network() -- in train_detail_functions (random init or load previous model)


-- Optimization part
losses_tr = 0 losses_te = 0

if opt.optim_method == 'rmsprop' then
  optim_state_diagRNN = {learningRate = opt.lRate0, alpha = opt.decay_rate}
  optim_state_uttRNN = {learningRate = opt.lRate0, alpha = opt.decay_rate}
elseif opt.optim_method == 'sgd' then
  optim_state_diagRNN =  {learningRate = opt.lRate0, momentum = opt.momentum, weightDecay = opt.wdcay}
  optim_state_uttRNN =  {learningRate = opt.lRate0, momentum = opt.momentum, weightDecay = opt.wdcay}
elseif opt.optim_method == 'adam' then
  optim_state_diagRNN =  {learningRate = opt.lRate0}
  optim_state_uttRNN =  {learningRate = opt.lRate0}
elseif opt.optim_method == 'lbfgs' then
  optim_state_diagRNN =  {learningRate = opt.lRate0}
  optim_state_uttRNN =  {learningRate = opt.lRate0}
elseif opt.optim_method == 'adadelta' then
  optim_state_diagRNN =  {learningRate = opt.lRate0}
  optim_state_uttRNN =  {learningRate = opt.lRate0}
end

local iter_per_epoch = math.ceil(loader.ntrain_diag/opt.batch_sz)
local iterations = iter_per_epoch * opt.maxEpoch

print('Start training')


-- DEBUG (test & valid)
--[[
print(string.format("\n Testing... please wait\n"))
losses_te, acc_te = test() -- Measure test accuracy
print('Done')

print(string.format("\n Validing... please wait\n"))
losses_val, acc_val = valid() -- Measure test accuracy
print('Done')
--]]


nUtt_processed_tr = 0
for iter=(opt.startEpoch-1)*iter_per_epoch+1, iterations do
  epoch = iter / iter_per_epoch

  if(epoch == opt.dRNN_only_epoch) then
    print('Change mode : dRNN only --> joint training')
    print('Optimization : adadelta --> sgd')
    torch.save(saveDir..'/protos_diag_dRNN_only', protos_diag)
    torch.save(saveDir..'/protos_utt_dRNN_only', protos_utt)
    opt.optim_method = 'sgd'
    optim_state_diagRNN =  {learningRate = opt.lRate0, momentum = opt.momentum, weightDecay = opt.wdcay}
    optim_state_uttRNN =  {learningRate = opt.lRate0, momentum = opt.momentum, weightDecay = opt.wdcay}
  end


  local iter_within_epoch = iter % iter_per_epoch
  local timer = torch.Timer()

  -- Forward & Backprop & Update

  if opt.optim_method == 'rmsprop' then
    _, loss = optim.rmsprop(feval_dRNN, params_dRNN, optim_state_diagRNN)
    if(epoch >=opt.dRNN_only_epoch) then
      _, _ = optim.rmsprop(feval_uRNN, params_uRNN, optim_state_uttRNN)
    end
  elseif opt.optim_method == 'sgd' then
    _, loss = optim.sgd(feval_dRNN, params_dRNN, optim_state_diagRNN)
    if(epoch >=opt.dRNN_only_epoch) then
      _, _ = optim.sgd(feval_uRNN, params_uRNN, optim_state_uttRNN)
    end
  elseif opt.optim_method == 'adam' then
    _, loss = optim.adam(feval_dRNN, params_dRNN, optim_state_diagRNN)
    if(epoch >=opt.dRNN_only_epoch) then
      _, _ = optim.adam(feval_uRNN, params_uRNN, optim_state_uttRNN)
    end
  elseif opt.optim_method == 'lbfgs' then
    _, loss = optim.lbfgs(feval_dRNN, params_dRNN, optim_state_diagRNN)
    if(epoch >=opt.dRNN_only_epoch) then
      _, _ = optim.lbfgs(feval_uRNN, params_uRNN, optim_state_uttRNN)
    end
  elseif opt.optim_method == 'adadelta' then
    _, loss = optim.adadelta(feval_dRNN, params_dRNN, optim_state_diagRNN)
    if(epoch >=opt.dRNN_only_epoch) then
      _, _ = optim.adadelta(feval_uRNN, params_uRNN, optim_state_uttRNN)
    end
  end

  local loss_tr = loss[1]

  losses_tr = losses_tr + loss[1] -- loss may not diverge (max double ~ 1e308)

  local time = timer:time().real

  -- Learning rate decay (later version)

  -- Display progress
  if iter % opt.print_every == 0 then
    acc_tr_percent = acc_tr/(nUtt_processed_tr)*100 -- This is invalid for last batch. but ignore it
    losses_tr_avg = losses_tr/(nUtt_processed_tr)

    if(epoch >=opt.dRNN_only_epoch) then
      print(string.format("epoch %.3f, l = %6.4f, a = %2.3f(pre=%d,tr=%d), g_dRnn = %6.3e, g_uRNN= %6.3e,  time/b = %.2fs ", epoch, losses_tr_avg,acc_tr_percent ,predIdx, target, grad_dRNN:norm(),  grad_uRNN:norm(),  time))
    else
      print(string.format("epoch %.3f, l = %6.4f, a = %2.3f(pre=%d,tr=%d), g_dRnn = %6.3e, p_dRNN = %6.3e,  time/b = %.2fs ", epoch, losses_tr_avg,acc_tr_percent ,predIdx, target, grad_dRNN:norm(), params_dRNN:norm(), time))
    end



  end

  if iter % 50 == 0 then collectgarbage() end  -- memory management

  -- Save result
  if iter % iter_per_epoch == 0 then
    loss_tr_epoch[epoch] = losses_tr/(loader.ntrain_utt)
    acc_tr_epoch[epoch] = acc_tr/(loader.ntrain_utt)*100
    print(string.format("\n Train result : epoch %.3f, loss_tr = %6.6f, acc_tr = %2.3f", epoch, loss_tr_epoch[epoch],acc_tr_epoch[epoch]))

    torch.save(saveDir..'/loss_tr_epoch', loss_tr_epoch)
    torch.save(saveDir..'/acc_tr_epoch', acc_tr_epoch)
    torch.save(saveDir..'/confusion_tr',confusion_tr)

    torch.save(saveDir..'/protos_diag', protos_diag)
    torch.save(saveDir..'/protos_utt', protos_utt)


    torch.save(saveDir..'/loss_each_tr',loss_each_tr)
    torch.save(saveDir..'/correct_each_tr',correct_each_tr)

    losses_tr = 0 -- Loss initialization
    confusion_tr:zero() -- Confusion matrix initialization
    acc_tr = 0 -- Training accuracy initialization
    nUtt_processed_tr = 0

    print(string.format("\n Testing... please wait\n"))
    losses_te, acc_te = test() -- Measure test accuracy
    loss_te_epoch[epoch] = losses_te
    acc_te_epoch[epoch] = acc_te
    print(string.format("Test result : epoch %.3f, loss_te = %6.6f, acc_te = %2.3f", epoch, loss_te_epoch[epoch],acc_te_epoch[epoch]))
    torch.save(saveDir..'/confusion_te',confusion_te)
    torch.save(saveDir..'/loss_te_epoch', loss_te_epoch)
    torch.save(saveDir..'/acc_te_epoch', acc_te_epoch)
    torch.save(saveDir..'/loss_each_te',loss_each_te)
    torch.save(saveDir..'/correct_each_te',correct_each_te)
    confusion_te:zero()

    print(string.format("\n Validing... please wait\n"))
    losses_val, acc_val = valid() -- Measure test accuracy
    loss_val_epoch[epoch] = losses_val
    acc_val_epoch[epoch] = acc_val
    print(string.format("Valid result : epoch %.3f, loss_val = %6.6f, acc_val = %2.3f", epoch, loss_val_epoch[epoch],acc_val_epoch[epoch]))
    torch.save(saveDir..'/confusion_val',confusion_val)
    torch.save(saveDir..'/loss_val_epoch', loss_val_epoch)
    torch.save(saveDir..'/acc_val_epoch', acc_val_epoch)
    torch.save(saveDir..'/loss_each_val',loss_each_val)
    torch.save(saveDir..'/correct_each_val',correct_each_val)
    confusion_val:zero()

    -- Save validmax model
    if(epoch > 1 and acc_val > maxval) then
      maxval = acc_val
      torch.save(saveDir..'/protos_diag_mv', protos_diag)
      torch.save(saveDir..'/protos_utt_mv',protos_utt)
    end

    -- Learning rate scheduling based on validation loss
    if(epoch > 3 and (loss_val_epoch[epoch] > loss_val_epoch[epoch-1])) then
      print('Validation loss increase. Halving lRate')
      optim_state_diagRNN.learningRate = optim_state_diagRNN.learningRate*0.8
      optim_state_uttRNN.learningRate = optim_state_uttRNN.learningRate*0.8

      print('learning Rate = '..optim_state_diagRNN.learningRate..' will be applied from next epoch')
    end
    print('Valid done')
  end
end


