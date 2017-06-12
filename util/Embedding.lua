local Embedding, parent = torch.class('Embedding', 'nn.Module')

function Embedding:__init(inputSize, outputSize)
  parent.__init(self)
  self.outputSize = outputSize
  self.weight = torch.Tensor(inputSize, outputSize)
  self.gradWeight = torch.Tensor(inputSize, outputSize)
end

function Embedding:updateOutput(input)
  self.output:resize(1, self.outputSize)
  self.output:copy(self.weight[input[1]])
  return self.output
end

function Embedding:updateGradInput(input, gradOutput)
  return self.gradInput
end

function Embedding:accGradParameters(input, gradOutput, scale)
  scale = scale or 1
  if scale == 0 then
    self.gradWeight:zero()
  end

  local word = input[1]
  self.gradWeight[word]:add(gradOutput)
end

-- we do not need to accumulate parameters when sharing
Embedding.sharedAccUpdateGradParameters = Embedding.accUpdateGradParameters
