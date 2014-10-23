

#include "harris.h"
#include <iostream>
#include <cv.h>
#include "math.h"
using namespace std;
using namespace cv;
#define MAX_CORNERS 200
#define countn 200
#define videofile "video1.avi"
#define ZERO 0
#define PI 3.1416
#define TrainNo 50
#define threshold1 1
#define threshold2 1
#define threshold3 1
#define threshold4 1

//定义结构体，存储速度 和角度
struct feature1
{
double speed;
double ang;
};



//排序函数，输出大的
bool biggerSort(vector<cv::Point> v1, vector<cv::Point> v2)
{
	return cv::contourArea(v1)>cv::contourArea(v2);
}

//计算密度
double den_cal(Mat forground)
{
	int number=countNonZero(forground);
	return double(number)/double((forground.rows*forground.cols));

}

//光流法，根据特征点，计算其运动速度及方向，存储在vector容器里面。
vector<struct feature1> detect_event(vector<cv::Point> corners,Mat cur_mat,Mat pre_mat)
{
	IplImage  img_grey_cur= IplImage(cur_mat);
	IplImage  img_grey_pre= IplImage(pre_mat); 
	vector<struct feature1> feature_value;
	feature_value.clear();
	//cout<<feature_value.max_size()<<endl;
	struct feature1 value;
	memset(&value,0,sizeof(value));
	CvPoint2D32f * move_new_point = new CvPoint2D32f[ MAX_CORNERS];
	CvPoint2D32f * move_old_point = new CvPoint2D32f[ MAX_CORNERS];
	//cout<<corners.size();
	for(int j=0;j<corners.size();j++)
		{
	move_old_point[j].x=corners[j].x;
	move_old_point[j].y=corners[j].y;
		}
	CvSize Pyrsize = cvSize(cur_mat.cols +8, cur_mat.rows/3);

	IplImage * pyrA = cvCreateImage(Pyrsize, IPL_DEPTH_32F, 1); //pyrA是需要寻找的点，不是没有初始化的
		
	IplImage * pyrB = cvCreateImage(Pyrsize, IPL_DEPTH_32F, 1);

	char *features_found = new char[MAX_CORNERS];

	float *features_error = new float[MAX_CORNERS];
	CvTermCriteria criteria;
	criteria = cvTermCriteria (CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 64, 0.01);

	cvCalcOpticalFlowPyrLK(&img_grey_pre,

		&img_grey_cur,

		pyrA,

		pyrB,

		move_old_point,//旧的特征点

		move_new_point,//要计算的新的特征点位置

		countn,

		cvSize(10, 10),

		3,

		features_found,

		features_error,

		criteria,

		0

		);

	double alpha_angle=0;
	for (int i = 0; i < countn; i++)

	{  
		memset(&value,0,sizeof(value));
		int x1 = (int)move_new_point[i].x;

		int x2 = (int)move_old_point[i].x;

		int y1 = (int)move_new_point[i].y;

		int y2 = (int)move_old_point[i].y;

		int dx =(int) abs(x1 - x2) ;

		int dy = (int)abs(y1 - y2);

		double len=dx*dx+dy*dy;
		int speed=std::sqrt(len);//计算速度
				
		if(dx==0)//计算角度
			alpha_angle = PI/2;
		else
			{double ang_value=dy/dx;
			alpha_angle = abs(std::atan(ang_value));
			}
		if(dx<0 && dy>0) alpha_angle = PI- alpha_angle ;
		if(dx<0 && dy<0) alpha_angle = PI + alpha_angle ;
		if(dx>0 && dy<0) alpha_angle = 2*PI - alpha_angle ;

		value.ang=alpha_angle;
		value.speed=speed;
		//cout<<"角度"<<value.ang<<"速度"<<value.speed<<endl;
		feature_value.push_back(value);

	}
	cvReleaseImage(&pyrA);
	cvReleaseImage(&pyrB);
	delete move_new_point;
	delete move_old_point;
	delete features_found;
	delete features_error;		
	return feature_value;//返回容器，存储的是一帧图片的运动矢量
}


//相似性判断
bool judge(vector<struct feature1>value,double density)
{	
	////求相似性
	double similarity=0;
	double simi_value=0;
	double motion_stren=0;
	//余弦相似度计算
	for(int i=0;i<value.size();i++)
	{	//cout<<value[i].ang<<endl;
		motion_stren=motion_stren+sqrt(value[i].ang*value[i].ang+value[i].speed*value[i].speed);
		for(int j=i+1;j<value.size();j++)
		{
			simi_value=(value[i].speed*value[j].speed+value[i].ang*value[j].ang)/\
			sqrt(value[i].speed*value[i].speed+value[i].ang*value[i].ang)*\
			sqrt(value[j].speed*value[j].speed+value[j].ang*value[j].ang);
			similarity=similarity+simi_value;
		}
	}
	motion_stren=motion_stren/value.size();
	similarity=2*similarity/(value.size()*value.size());
	double count=0;
	for(int j=0;j<value.size();j++)
	{
		double stren=sqrt(value[j].ang*value[j].ang+value[j].speed*value[j].speed);
		count=count+abs(stren-motion_stren)*abs(stren-motion_stren);
	}
	count=sqrt(count/value.size());
	cout<<"平均密度"<<density<<"平均运动强度"<<motion_stren<<"平均相似度"<<similarity<<"强度方差"<<count<<endl;
	if(density>threshold1&&motion_stren>threshold2&&similarity>threshold3)
		return true;
	else
		return false;
	
}


Mat change(Mat cur_mat)
{

	int result[8];
	for(int i=1;i<cur_mat.rows-1;i++)
		for(int j=1;j<cur_mat.cols-1;j++)
		{
			int sum=0;
			if(cur_mat.at<uchar>(i,j)>cur_mat.at<uchar>(i,j-1))
				result[0]=0;
			else
				result[0]=1;
			if(cur_mat.at<uchar>(i,j)>cur_mat.at<uchar>(i-1,j-1))
				result[1]=0;
			else
				result[1]=1;
			if(cur_mat.at<uchar>(i,j)>cur_mat.at<uchar>(i-1,j))
				result[2]=0;
			else
				result[2]=1;
			if(cur_mat.at<uchar>(i,j)>cur_mat.at<uchar>(i-1,j+1))
				result[3]=0;
			else
				result[3]=1;
			if(cur_mat.at<uchar>(i,j)>cur_mat.at<uchar>(i,j+1))
				result[4]=0;
			else
				result[4]=1;
			if(cur_mat.at<uchar>(i,j)>cur_mat.at<uchar>(i+1,j+1))
				result[5]=0;
			else
				result[5]=1;
			if(cur_mat.at<uchar>(i,j)>cur_mat.at<uchar>(i+1,j))
				result[6]=0;
			else
				result[6]=1;
			if(cur_mat.at<uchar>(i,j)>cur_mat.at<uchar>(i+1,j-1))
				result[7]=0;
			else
				result[7]=1;

			for(int k=0;k<8;k++)
			{
				float z=(double)pow((float)2,(float)k);
				sum=sum+z*result[k];
			}
			cur_mat.at<uchar>(i,j)=sum;

		}
		return cur_mat;
}

int main()
{

Mat pre_image,pre_mat;
Mat cur_image,cur_mat;
Mat change_mat;
cv::VideoCapture capture;
capture.open(videofile);
std::vector<cv::Point> corners;

vector<struct feature1> value;
int frameno=capture.get(CV_CAP_PROP_FRAME_COUNT);
int i=0;
cv::BackgroundSubtractorMOG2 mog;	
cv::Mat frame;			//当前帧
cv::Mat background;
cv::Mat foreground;		//前景
cv::Mat bw;				//中间二值变量
cv::Mat se;				//形态学结构元素
for(i=0;i<TrainNo;++i)
{
	cout<<"正在训练背景:"<<i<<endl;
	capture.read(frame);
	cvtColor(frame,frame,CV_RGB2GRAY);

	if(frame.empty()==true)
	{
		cout<<"视频帧太少，无法训练背景"<<endl;
		getchar();
		return 0;
	}
	mog(frame,foreground,0.01);	
}

cv::Rect rt;
se=cv::getStructuringElement(cv::MORPH_RECT,cv::Size(5,5));
//统计目标直方图时使用到的变量
vector<cv::Mat> vecImg;
vector<int> vecChannel;
vector<int> vecHistSize;
vector<float> vecRange;
cv::Mat mask(frame.rows,frame.cols,cv::DataType<uchar>::type);
//变量初始化
vecChannel.push_back(0);
vecHistSize.push_back(32);
vecRange.push_back(0);
vecRange.push_back(180);
cv::MatND hist;		//直方图数组
double maxVal;		//直方图最大值，为了便于投影图显示，需要将直方图规一化到[0 255]区间上
cv::Mat backP;		//反射投影图
cv::Mat result;		//跟踪结果
cv::Mat hsv;
capture.read(pre_image);
cv::cvtColor (pre_image,pre_mat,CV_BGR2GRAY);
pre_mat.convertTo(pre_mat,CV_8UC1);
while(i<frameno)
	{

capture.read(cur_image);
cur_image.copyTo(result);
cv::cvtColor(cur_image,cur_mat,CV_BGR2GRAY);
change_mat=change(cur_mat);
mog(cur_image,foreground,0.01);

mog.getBackgroundImage(background);
//cout<<background.channels();
cv::cvtColor(frame,hsv,cv::COLOR_BGR2HSV);
//对前景进行中值滤波、形态学膨胀操作，以去除伪目标和接连断开的小目标（一个大车辆有时会断开成几个小目标）	
cv::medianBlur(foreground,foreground,5);
//cv::imshow("中值滤波",foreground);
//cvMoveWindow("中值滤波",800,0);
cv::morphologyEx(foreground,foreground,cv::MORPH_DILATE,se);//形态学运算

double density=den_cal(foreground);///////////////////////////////////////density

//检索前景中各个连通分量的轮廓
foreground.copyTo(bw);
vector<vector<cv::Point>> contours;
cv::findContours(bw,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);
if(contours.size()<1)
	continue;
//对连通分量进行排序
std::sort(contours.begin(),contours.end(),biggerSort);

//结合camshift更新跟踪位置（由于camshift算法在单一背景下，跟踪效果非常好；
//但是在监控视频中，由于分辨率太低、视频质量太差、目标太大、目标颜色不够显著
//等各种因素，导致跟踪效果非常差。  因此，需要边跟踪、边检测，如果跟踪不够好，
//就用检测位置修改
vecImg.clear();
vecImg.push_back(hsv);
for(int k=0;k<contours.size();++k)
{
	//第k个连通分量的外接矩形框
	if(cv::contourArea(contours[k])<cv::contourArea(contours[0])/5)
		break;
	rt=cv::boundingRect(contours[k]);				
	mask=0;
	mask(rt)=255;
	//统计直方图
	cv::calcHist(vecImg,vecChannel,mask,hist,vecHistSize,vecRange);				
	cv::minMaxLoc(hist,0,&maxVal);
	hist=hist*255/maxVal;
	//计算反向投影图
	cv::calcBackProject(vecImg,vecChannel,hist,backP,vecRange,1);
	//camshift跟踪位置
	cv::Rect search=rt;
	cv::RotatedRect rrt=cv::CamShift(backP,search,cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,10,1));
	cv::Rect rt2=rrt.boundingRect();
	rt&=rt2;

	//跟踪框画到视频上
	cv::rectangle(result,rt,cv::Scalar(0,255,0),2);	

}

cv::imshow("跟踪效果",result);
cvMoveWindow("跟踪效果",500,150);
cvWaitKey(15);
// 计算角点，计算特征点  
cv::goodFeaturesToTrack(cur_mat, corners,  
	200,  
	//角点最大数目  
	0.01,  
	// 质量等级，这里是0.01*max（min（e1，e2）），e1，e2是harris矩阵的特征值  
	10);  
value=detect_event(corners,cur_mat,pre_mat);//调用运动矢量计算函数
bool judge_result=judge(value,density);//调用求特征函数
value.clear();
if(judge_result)//响应部分
{
Beep(500,500);
}


capture.read(pre_image);//读取下一帧
cv::cvtColor (pre_image,pre_mat,CV_BGR2GRAY);
pre_mat.convertTo(pre_mat,CV_8UC1);
i=i+2;
//i=i+1
//pre_mat=cur_mat;

}
cv::waitKey (0);  
return 0;
}