class Config:
    """ For holding model hyperparams """

    #def __init__(self,
    #             batch_size=100,
    #             embed_size=100,
    #             hidden_size=50,
    #             steps=10,
    #             max_epochs=30,
    #             early_stopping=30,
    #             dropout=0.5,
    #             lr=0.0001,
    #             l2=0.1):
    #    self.batch_size = batch_size
    #    self.embed_size = embed_size
    #    self.hidden_size = hidden_size
    #    self.steps = steps
    #    self.max_epochs = max_epochs
    #    self.early_stopping = early_stopping
    #    self.dropout = dropout
    #    self.lr = lr
    #    self.l2 = l2


    def __init__(self,mode=''):
        if mode == 'LSTM':
            self.batch_size = 64
            self.embed_size = 64
            self.hidden_size = 50
            self.steps = 10
            self.max_epochs = 30
            self.early_stopping = 30
            self.dropout = 0.5
            self.lr = 0.0001
            self.l2 = 0.1
        elif mode == 'CNN':
            self.batch_size = 100
            self.embed_size = 100
            self.hidden_size = 50
            self.steps = 10
            self.max_epochs = 30
            self.early_stopping = 30
            self.dropout = 0.5
            self.lr = 0.0001
            self.l2 = 0.1
        elif mode == 'RNN':
            self.batch_size = 100
            self.embed_size = 100
            self.hidden_size = 50
            self.steps = 10
            self.max_epochs = 30
            self.early_stopping = 30
            self.dropout = 0.5
            self.lr = 0.0001
            self.l2 = 0.1
        else:
            self.batch_size = 100
            self.embed_size = 100
            self.hidden_size = 50
            self.steps = 20
            self.max_epochs = 30
            self.early_stopping = 30
            self.dropout = 0.5
            self.lr = 0.0001
            self.l2 = 0.1



