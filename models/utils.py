def show_params(nnet):
    print("=" * 40, "Model Parameters", "=" * 40)
    num_params = 0
    for module_name, m in nnet.named_modules():
        if module_name == '':
            for name, params in m.named_parameters():
                print(name, params.size())
                i = 1
                for j in params.size():
                    i = i * j
                num_params += i

    print('[*] Parameter Size: {}'.format(num_params))
    print("=" * 100)


def show_model(nnet):
    print("=" * 40, "Model Structures", "=" * 40)
    for module_name, m in nnet.named_modules():
        if module_name == '':
            print(m)

    print("=" * 100)


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())

    return num_param