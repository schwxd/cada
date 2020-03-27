import torch

a = torch.tensor([1., 2., 3.], requires_grad=True)

print(a.grad)

# out = a.sigmoid()
out = a.relu()
print(out)

# c = out.detach()
c = out.data
print(c)
# c.sum().backward()
# c.zero_()

out.sum().backward()

print(a.grad)
print(a)