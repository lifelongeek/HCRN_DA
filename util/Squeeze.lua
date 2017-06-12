local Squeeze, parent = torch.class('nn.Squeeze', 'nn.Module')

function Squeeze:updateOutput(input)
  
  self.size = input:size()
  --self.output = input:squeeze()   
  output = input:squeeze(4)
  self.output = output:squeeze(3)
  return self.output
end

function Squeeze:updateGradInput(input, gradOutput)
  self.gradInput = gradOutput:view(self.size)
  return self.gradInput  
end
