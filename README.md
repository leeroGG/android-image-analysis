# android-image-analysis
OpenCV + Tensorflow 配合应用，实现图像实时分析

事实上实现并不困难，毕竟OpenCV还是很强大的，而且使用起来也是很方便。

使用OpenCV调取到摄像头画面，并且能对图像进行修改，这就完成一半了。

接着是找一个训练好的算法模型（可以使用Tensorflow开源的），把OpenCV的图像容器Mat对象转换成Bitmap对象，使用Tensorflow调用算法模型进行分析，最后把分析结果显示出来就完事了。

虽然文章写得不咋滴，还是放放，哈哈哈</br>
[https://www.jianshu.com/p/76c9eed732a5](https://www.jianshu.com/p/76c9eed732a5)
