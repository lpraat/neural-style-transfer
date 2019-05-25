import tensorflow as tf
import numpy as np
import cv2
import time

IMAGENET_MEAN = np.array([103.939, 116.779, 123.68])

def create_model(input_shape, content_layer, style_layers):
    vgg19 = tf.keras.applications.vgg19.VGG19(weights='imagenet', include_top=False)
    target_layers = [content_layer] + style_layers
    return tf.keras.models.Model(
        inputs=vgg19.input,
        outputs=[vgg19.get_layer(layer).output for layer in target_layers]
    )

def get_and_preprocess(img_path, size):
    img = cv2.imread(img_path)
    img = cv2.resize(img, size)
    img = img.astype(np.float32)
    img -= IMAGENET_MEAN
    return img.reshape(1, *img.shape)

def unprocess(img):
    img += IMAGENET_MEAN
    img = np.clip(img, 0, 255).astype('uint8')
    return img

def content_cost(out_activations, content_activations):
    return tf.reduce_mean(tf.square(out_activations - content_activations))

def _style_cost(out_activations, style_activations):
    _, width, height, channels = out_activations.get_shape()
    out_activations = tf.reshape(tf.transpose(out_activations), [channels, width*height])
    style_activations = tf.reshape(tf.transpose(style_activations), [channels, width*height])

    gram_style_out = tf.matmul(out_activations, out_activations, transpose_b=True)
    gram_style_target = tf.matmul(style_activations, style_activations, transpose_b=True)

    return tf.reduce_mean(tf.square(gram_style_target - gram_style_out))

def style_cost(outs_activations, style_activations):
    weight = 1 / len(style_activations)

    cost = 0
    for i in range(len(style_activations)):
        cost += weight * _style_cost(outs_activations[i], style_activations[i])

    return cost

def generate_image(
      content_path,
      style_path,
      out_path="./out.jpg",
      size=(320, 240),
      content_layer="block5_conv2",
      content_weight=1e3,
      style_layers=['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'],
      styel_weight=1e-3,
      learning_rate=1.0,
      num_iterations=1000,
      save_every=50
    ):

    content_image = get_and_preprocess(content_path, size)
    style_image = get_and_preprocess(style_path, size)

    model = create_model(content_image.shape, content_layer, style_layers)

    # To see model architeture
    # print model.summary()

    # Precompute target content activations
    content_activations = model.predict(content_image)[0]

    # Precompute target style activations
    style_activations = model.predict(style_image)[1:]

    # Start generating from the content image (rather than noise)
    out_image = tf.Variable(content_image, dtype=tf.float32)

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    @tf.function
    def train_step():
      with tf.GradientTape() as tape:
        outputs = model(out_image)
        cost = content_weight * content_cost(outputs[0], content_activations) + \
               styel_weight * style_cost(outputs[1:], style_activations)

      grads = tape.gradient(cost, out_image)
      opt.apply_gradients([(grads, out_image)])
      return cost

    start = time.time()
    for i in range(1, num_iterations + 1):
        cost = train_step()

        if i % save_every == 0:
            print(f"Done {i} iterations. Took {time.time()-start :.2f} seconds")
            print(f"Current cost {cost}")
            print("Saving generated image to file...\n")

            out = out_image.numpy().reshape(*out_image.shape[1:])
            cv2.imwrite(out_path, unprocess(out))

            start = time.time()