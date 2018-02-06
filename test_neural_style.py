from neural_style import NSTModel

# Need to specify the content image path and the style image path
nst = NSTModel(content_path='img/bridge.jpg', style_path='img/p1.jpg', h=768, w=1024, style_weights=[0.5, 1, 2, 3, 4])
nst.run(num_iter=2000, beta=1)

