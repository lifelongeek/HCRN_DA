
local OneHot, parent = torch.class('OneHot', 'nn.Module')

function OneHot:__init(outputSize)
  parent.__init(self)
  self.outputSize = outputSize
  -- We'll construct one-hot encodings by using the index method to
  -- reshuffle the rows of an identity matrix. To avoid recreating
  -- it every iteration we'll cache it.
  self._eye = torch.eye(outputSize)
end

function OneHot:updateOutput(input) 
  -- self.output:resize(input:size(1), self.outputSize):zero()
  self.output:resize(1, self.outputSize):zero()   -- Assumption : One utterance at a time --by GGM
  -- print(type(input))
  if self._eye == nil then self._eye = torch.eye(self.outputSize) end
  self._eye = self._eye:float()
  local longInput = torch.LongTensor(1) --input:long()  --by GGM
  longInput[1] = input --by GGM
  self.output:copy(self._eye:index(1, longInput))
  -- print(#self.output) --by GGM
  return self.output
end
