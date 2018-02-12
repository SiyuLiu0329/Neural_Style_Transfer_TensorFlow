from neural_style import NSTModel

# Need to specify the content image path and the style image path
nst = NSTModel(content_path='img/content.jpg', style_path='img/style.jpg', h=768, w=1024)
nst.run(num_iter=200)

