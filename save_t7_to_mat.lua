require 'optim'

cmd = torch.CmdLine()

cmd:option('-expnum',0,'experiment number')
opt = cmd:parse(arg)

print('Experiment number = '..opt.expnum)

local nOutput = 42

load_dir = 'exp/'..opt.expnum

loss_tr_epoch = torch.load(load_dir..'/loss_tr_epoch')
loss_te_epoch = torch.load(load_dir..'/loss_te_epoch')
loss_val_epoch = torch.load(load_dir..'/loss_val_epoch')
acc_tr_epoch = torch.load(load_dir..'/acc_tr_epoch')
acc_te_epoch = torch.load(load_dir..'/acc_te_epoch')
acc_val_epoch = torch.load(load_dir..'/acc_val_epoch')

--[[
confusion_tr = torch.load(load_dir..'/confusion_tr')
confusion_te = torch.load(load_dir..'/confusion_te')
--]]


loss_each_tr = torch.load(load_dir..'/loss_each_tr')
loss_each_te = torch.load(load_dir..'/loss_each_te')
loss_each_val = torch.load(load_dir..'/loss_each_val')
correct_each_tr = torch.load(load_dir..'/correct_each_tr')
correct_each_te = torch.load(load_dir..'/correct_each_te')
correct_each_val = torch.load(load_dir..'/correct_each_val')

confusion_tr_table = torch.load(load_dir..'/confusion_tr')
confusion_te_table = torch.load(load_dir..'/confusion_te')
confusion_val_table = torch.load(load_dir..'/confusion_val')

confusion_tr = torch.Tensor(nOutput,nOutput)
confusion_te = torch.Tensor(nOutput,nOutput)
confusion_val = torch.Tensor(nOutput,nOutput)
for i=1,nOutput do
for j=1,nOutput do
	confusion_tr[i][j] = confusion_tr_table.mat[i][j]
	confusion_te[i][j] = confusion_te_table.mat[i][j]
	confusion_val[i][j] = confusion_val_table.mat[i][j]
end
end


-- mattorch = require 'mattorch'\
matio = require 'matio'

matio.save(load_dir..'/loss_tr_epoch.mat',loss_tr_epoch)
matio.save(load_dir..'/loss_te_epoch.mat',loss_te_epoch)
matio.save(load_dir..'/loss_val_epoch.mat',loss_val_epoch)
matio.save(load_dir..'/acc_tr_epoch.mat',acc_tr_epoch)
matio.save(load_dir..'/acc_te_epoch.mat',acc_te_epoch)
matio.save(load_dir..'/acc_val_epoch.mat',acc_val_epoch)
matio.save(load_dir..'/loss_each_tr.mat',loss_each_tr)
matio.save(load_dir..'/loss_each_te.mat',loss_each_te)
matio.save(load_dir..'/loss_each_val.mat',loss_each_val)
matio.save(load_dir..'/correct_each_tr.mat',correct_each_tr)
matio.save(load_dir..'/correct_each_te.mat',correct_each_te)
matio.save(load_dir..'/correct_each_val.mat',correct_each_val)
matio.save(load_dir..'/confusion_tr.mat',confusion_tr)
matio.save(load_dir..'/confusion_te.mat',confusion_te)
matio.save(load_dir..'/confusion_val.mat',confusion_val)

--[[
mattorch.save(load_dir..'/loss_tr_epoch.mat',loss_tr_epoch)
mattorch.save(load_dir..'/loss_te_epoch.mat',loss_te_epoch)
mattorch.save(load_dir..'/loss_val_epoch.mat',loss_val_epoch)
mattorch.save(load_dir..'/acc_tr_epoch.mat',acc_tr_epoch)
mattorch.save(load_dir..'/acc_te_epoch.mat',acc_te_epoch)
mattorch.save(load_dir..'/acc_val_epoch.mat',acc_val_epoch)
mattorch.save(load_dir..'/loss_each_tr.mat',loss_each_tr)
mattorch.save(load_dir..'/loss_each_te.mat',loss_each_te)
mattorch.save(load_dir..'/loss_each_val.mat',loss_each_val)
mattorch.save(load_dir..'/correct_each_tr.mat',correct_each_tr)
mattorch.save(load_dir..'/correct_each_te.mat',correct_each_te)
mattorch.save(load_dir..'/correct_each_val.mat',correct_each_val)
mattorch.save(load_dir..'/confusion_tr.mat',confusion_tr)
mattorch.save(load_dir..'/confusion_te.mat',confusion_te)
mattorch.save(load_dir..'/confusion_val.mat',confusion_val)
--]]

