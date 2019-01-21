package com.smartahc.android.camera

import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.os.Looper
import android.os.MessageQueue
import android.support.v7.app.AppCompatActivity
import com.smartahc.android.camera.ternsorflow.Classifier
import com.smartahc.android.camera.ternsorflow.TensorFlowImageClassifier
import kotlinx.android.synthetic.main.activity_camera.*
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import java.io.IOException
import java.util.concurrent.Executor
import java.util.concurrent.ScheduledThreadPoolExecutor
import java.util.concurrent.ThreadFactory
import org.opencv.core.Mat




class CameraActivity : AppCompatActivity(), CameraBridgeViewBase.CvCameraViewListener {

    private var grayscaleImage: Mat? = null
    private var absoluteFaceSize: Int = 0

    private val INPUT_SIZE = 224
    private val IMAGE_MEAN = 117
    private val IMAGE_STD = 1f
    private val INPUT_NAME = "input"
    private val OUTPUT_NAME = "output"
    private val MODEL_FILE = "file:///android_asset/model/tensorflow_inception_graph.pb"
    private val LABEL_FILE = "file:///android_asset/model/imagenet_comp_graph_label_strings.txt"

    private var executor: Executor? = null
    private var classifier: Classifier? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_camera)

        cameraView.setCameraIndex(0) // 0:后置 1:前置
        cameraView.enableFpsMeter() //显示FPS
        cameraView.setCvCameraViewListener(this)

        cameraView.enableView()

        // 避免耗时任务占用 CPU 时间片造成UI绘制卡顿，提升启动页面加载速度
        Looper.myQueue().addIdleHandler(idleHandler)
    }

    /**
     * 主线程消息队列空闲时（视图第一帧绘制完成时）处理耗时事件
     */
    private var idleHandler: MessageQueue.IdleHandler = MessageQueue.IdleHandler {
        if (classifier == null) {
            // 创建 Classifier
            classifier = TensorFlowImageClassifier.create(this@CameraActivity.assets,
                    MODEL_FILE, LABEL_FILE, INPUT_SIZE, IMAGE_MEAN, IMAGE_STD, INPUT_NAME, OUTPUT_NAME)
        }

        // 初始化线程池
        executor = ScheduledThreadPoolExecutor(1, ThreadFactory { r ->
            val thread = Thread(r)
            thread.isDaemon = true
            thread.name = "ThreadPool-ImageClassifier"
            thread
        })

        false
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        grayscaleImage = Mat(height, width, CvType.CV_8UC4)
        absoluteFaceSize = (height * 0.2).toInt()
    }

    override fun onCameraViewStopped() {
        grayscaleImage?.release()
    }

    override fun onCameraFrame(inputFrame: Mat): Mat {
        Imgproc.cvtColor(inputFrame, grayscaleImage, Imgproc.COLOR_RGBA2RGB)

        /*val random = Math.random() * 100 + 1
        Imgproc.rectangle(inputFrame, Point(random, random), Point(random + 30, random + 30), Scalar(0.0, 255.0, 0.0, 255.0))*/

        executor?.execute {
            val bmpCanny = Bitmap.createBitmap(inputFrame.cols(), inputFrame.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(inputFrame, bmpCanny)
            val croppedBitmap = getScaleBitmap(bmpCanny, INPUT_SIZE)
            val results = classifier?.recognizeImage(croppedBitmap)
            runOnUiThread {
                tvInfo.text = results.toString()
            }
        }
//        Imgproc.putText(inputFrame, results.toString(), Point(0.0, 30.0), Core.FONT_HERSHEY_COMPLEX, 1.0, Scalar(255.0))

        return inputFrame
    }

    /**
     * 对图片进行缩放
     * @param bitmap
     * @param size
     * @return
     * @throws IOException
     */
    @Throws(IOException::class)
    private fun getScaleBitmap(bitmap: Bitmap, size: Int): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        val scaleWidth = size.toFloat() / width
        val scaleHeight = size.toFloat() / height
        val matrix = Matrix()
        matrix.postScale(scaleWidth, scaleHeight)
        return Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, true)
    }
}
