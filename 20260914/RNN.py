'''

    embedding -> RNN
        input_size: (batch_size, sequence_length, embedding_dim)

        
    output
        shape: (batch_size, sequence_length, hidden_size)
    
        
    hidden
        shape: (num_layers * num_directions, batch_size, hidden_size)

'''




"""
    PROBLEM:
    1.长期依赖（梯度消失）
        --> LSTM  GRU

    2.串行化（效率低）
        --> Transformer

    
"""