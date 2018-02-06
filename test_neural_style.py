from neural_style import NSTModel

nsf = NSTModel(content_path='img/bridge.jpg', style_path='img/ab1.jpg', h=768, w=1024, style_weights=[0.5, 1, 2, 3, 4])
nsf.run(num_iter=2000, beta=1)
