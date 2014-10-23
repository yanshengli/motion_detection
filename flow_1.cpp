

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

//����ṹ�壬�洢�ٶ� �ͽǶ�
struct feature1
{
double speed;
double ang;
};



//��������������
bool biggerSort(vector<cv::Point> v1, vector<cv::Point> v2)
{
	return cv::contourArea(v1)>cv::contourArea(v2);
}

//�����ܶ�
double den_cal(Mat forground)
{
	int number=countNonZero(forground);
	return double(number)/double((forground.rows*forground.cols));

}

//�����������������㣬�������˶��ٶȼ����򣬴洢��vector�������档
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

	IplImage * pyrA = cvCreateImage(Pyrsize, IPL_DEPTH_32F, 1); //pyrA����ҪѰ�ҵĵ㣬����û�г�ʼ����
		
	IplImage * pyrB = cvCreateImage(Pyrsize, IPL_DEPTH_32F, 1);

	char *features_found = new char[MAX_CORNERS];

	float *features_error = new float[MAX_CORNERS];
	CvTermCriteria criteria;
	criteria = cvTermCriteria (CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 64, 0.01);

	cvCalcOpticalFlowPyrLK(&img_grey_pre,

		&img_grey_cur,

		pyrA,

		pyrB,

		move_old_point,//�ɵ�������

		move_new_point,//Ҫ������µ�������λ��

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
		int speed=std::sqrt(len);//�����ٶ�
				
		if(dx==0)//����Ƕ�
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
		//cout<<"�Ƕ�"<<value.ang<<"�ٶ�"<<value.speed<<endl;
		feature_value.push_back(value);

	}
	cvReleaseImage(&pyrA);
	cvReleaseImage(&pyrB);
	delete move_new_point;
	delete move_old_point;
	delete features_found;
	delete features_error;		
	return feature_value;//�����������洢����һ֡ͼƬ���˶�ʸ��
}


//�������ж�
bool judge(vector<struct feature1>value,double density)
{	
	////��������
	double similarity=0;
	double simi_value=0;
	double motion_stren=0;
	//�������ƶȼ���
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
	cout<<"ƽ���ܶ�"<<density<<"ƽ���˶�ǿ��"<<motion_stren<<"ƽ�����ƶ�"<<similarity<<"ǿ�ȷ���"<<count<<endl;
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
cv::Mat frame;			//��ǰ֡
cv::Mat background;
cv::Mat foreground;		//ǰ��
cv::Mat bw;				//�м��ֵ����
cv::Mat se;				//��̬ѧ�ṹԪ��
for(i=0;i<TrainNo;++i)
{
	cout<<"����ѵ������:"<<i<<endl;
	capture.read(frame);
	cvtColor(frame,frame,CV_RGB2GRAY);

	if(frame.empty()==true)
	{
		cout<<"��Ƶ̫֡�٣��޷�ѵ������"<<endl;
		getchar();
		return 0;
	}
	mog(frame,foreground,0.01);	
}

cv::Rect rt;
se=cv::getStructuringElement(cv::MORPH_RECT,cv::Size(5,5));
//ͳ��Ŀ��ֱ��ͼʱʹ�õ��ı���
vector<cv::Mat> vecImg;
vector<int> vecChannel;
vector<int> vecHistSize;
vector<float> vecRange;
cv::Mat mask(frame.rows,frame.cols,cv::DataType<uchar>::type);
//������ʼ��
vecChannel.push_back(0);
vecHistSize.push_back(32);
vecRange.push_back(0);
vecRange.push_back(180);
cv::MatND hist;		//ֱ��ͼ����
double maxVal;		//ֱ��ͼ���ֵ��Ϊ�˱���ͶӰͼ��ʾ����Ҫ��ֱ��ͼ��һ����[0 255]������
cv::Mat backP;		//����ͶӰͼ
cv::Mat result;		//���ٽ��
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
//��ǰ��������ֵ�˲�����̬ѧ���Ͳ�������ȥ��αĿ��ͽ����Ͽ���СĿ�꣨һ��������ʱ��Ͽ��ɼ���СĿ�꣩	
cv::medianBlur(foreground,foreground,5);
//cv::imshow("��ֵ�˲�",foreground);
//cvMoveWindow("��ֵ�˲�",800,0);
cv::morphologyEx(foreground,foreground,cv::MORPH_DILATE,se);//��̬ѧ����

double density=den_cal(foreground);///////////////////////////////////////density

//����ǰ���и�����ͨ����������
foreground.copyTo(bw);
vector<vector<cv::Point>> contours;
cv::findContours(bw,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);
if(contours.size()<1)
	continue;
//����ͨ������������
std::sort(contours.begin(),contours.end(),biggerSort);

//���camshift���¸���λ�ã�����camshift�㷨�ڵ�һ�����£�����Ч���ǳ��ã�
//�����ڼ����Ƶ�У����ڷֱ���̫�͡���Ƶ����̫�Ŀ��̫��Ŀ����ɫ��������
//�ȸ������أ����¸���Ч���ǳ��  ��ˣ���Ҫ�߸��١��߼�⣬������ٲ����ã�
//���ü��λ���޸�
vecImg.clear();
vecImg.push_back(hsv);
for(int k=0;k<contours.size();++k)
{
	//��k����ͨ��������Ӿ��ο�
	if(cv::contourArea(contours[k])<cv::contourArea(contours[0])/5)
		break;
	rt=cv::boundingRect(contours[k]);				
	mask=0;
	mask(rt)=255;
	//ͳ��ֱ��ͼ
	cv::calcHist(vecImg,vecChannel,mask,hist,vecHistSize,vecRange);				
	cv::minMaxLoc(hist,0,&maxVal);
	hist=hist*255/maxVal;
	//���㷴��ͶӰͼ
	cv::calcBackProject(vecImg,vecChannel,hist,backP,vecRange,1);
	//camshift����λ��
	cv::Rect search=rt;
	cv::RotatedRect rrt=cv::CamShift(backP,search,cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,10,1));
	cv::Rect rt2=rrt.boundingRect();
	rt&=rt2;

	//���ٿ򻭵���Ƶ��
	cv::rectangle(result,rt,cv::Scalar(0,255,0),2);	

}

cv::imshow("����Ч��",result);
cvMoveWindow("����Ч��",500,150);
cvWaitKey(15);
// ����ǵ㣬����������  
cv::goodFeaturesToTrack(cur_mat, corners,  
	200,  
	//�ǵ������Ŀ  
	0.01,  
	// �����ȼ���������0.01*max��min��e1��e2������e1��e2��harris���������ֵ  
	10);  
value=detect_event(corners,cur_mat,pre_mat);//�����˶�ʸ�����㺯��
bool judge_result=judge(value,density);//��������������
value.clear();
if(judge_result)//��Ӧ����
{
Beep(500,500);
}


capture.read(pre_image);//��ȡ��һ֡
cv::cvtColor (pre_image,pre_mat,CV_BGR2GRAY);
pre_mat.convertTo(pre_mat,CV_8UC1);
i=i+2;
//i=i+1
//pre_mat=cur_mat;

}
cv::waitKey (0);  
return 0;
}