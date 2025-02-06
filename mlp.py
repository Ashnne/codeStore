class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, 
                input_dim:int, 
                output_dim:int, 
                middle_dim:[int],
                stddev:float,
                activation=None, # todo
                norm=None):      # todo
        super().__init__()

        input_dims = [input_dim, *middle_dim]
        output_dims = [*middle_dim, output_dim]

        layers = []

        for input_dim, output_dim in zip(input_dims,output_dims):
            layers.append(torch.nn.Linear(input_dim,output_dim))
            layers.append(torch.nn.ReLU())
        
        self.network = torch.nn.Sequential(*layers)
        self.init_weight(stddev)

    def init_weight(self,stddev):

        for i in self.network:
            if isinstance(i,torch.nn.Linear):
                torch.nn.init.uniform_(i.weight, -stddev, stddev)
                torch.nn.init.zeros_(i.bias)
    
    def forward(self,x):
        return self.network(x)