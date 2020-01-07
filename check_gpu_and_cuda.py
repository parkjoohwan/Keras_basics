import tensorflow as tf

# True / True가 출력되면 정상 작동중

print(tf.test.is_built_with_cuda())
print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))