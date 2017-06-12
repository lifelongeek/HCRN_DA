
-- misc utilities


function clone_list(tensor_list, zero_too) -- takes a list of tensors and returns a list of cloned tensors
  local out = {}
  if(tensor_list==nil) then tensor_list = {} end
  for k,v in pairs(tensor_list) do
    out[k] = v:clone()
    if zero_too then out[k]:zero() end
  end
  return out
end
