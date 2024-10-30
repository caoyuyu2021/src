%3）目标分解（分解经过验证的目标）

% 整车参数
G     = 2216.1;      %整车半载重量（Kg）
G_f   = 1108.1;      %半载前轴荷（Kg）
G_r   = G - G_f;     %半载后轴荷（Kg）
wheel_track_f  = 1580.3;  %轮距（mm）
wheel_track_r  = 1580.3;  %轮距（mm）
vehicle_height = 1800;    %车辆高度（mm）
area_aero = (wheel_track_f+wheel_track_r)/2*vehicle_height*10^-6; %迎风面积估算，m^2

total_roll_grad_t  = 4.500;  %选取一个侧倾梯度，deg/g
Uc = 60/3.6;      %输入车速
dirve_f=0.5;      %前轮驱动力占比

% 四轮定位参数
% nk    =  39.000; %主销拖距，输入参数，mm
nr_f  =  30.000; %轮胎拖距，设计参数，mm
nr_r  =  30.000; %轮胎拖距，设计参数，mm
toe_f =  0.0;  %设计状态下前束角，deg
toe_r =  0.12;  %设计状态下前束角，deg

%悬架KC参数
align_comp_f =  1.4100E-3;  %回正柔度系数，deg/Nm，发动机开
align_comp_r =  8.576500E-4;   %回正柔度系数，deg/Nm
fy_steer_f   = -0.03E-3;  %侧向力转向系数，deg/N
fy_steer_r   =  0.0305E-3;   %侧向力转向系数，deg/N
fx_steer_f   = -0.1277E-3;  %前轮驱动力转向系数，deg/N
fx_steer_r   = -0.0837E-3;  %后驱动力转向系数，deg/N
roll_steer_f = -6.99E-2;   %前悬架侧倾转向梯度，deg/deg
roll_steer_r =  4.83E-2;   %后悬架侧倾转向梯度，deg/deg
roll_camber_f = -0.8922;    %前悬架侧倾外倾梯度，deg/deg
roll_camber_r = -0.6625;    %后悬架侧倾外倾梯度，deg/deg
fy_camber_f= -0.12e-3;      %前悬架侧向力外倾梯度，deg/N
fy_camber_r= -0.18e-3;      %后悬架侧向力外倾梯度，deg/N


%轮胎导入的侧偏柔度/侧倾刚度
tire_corner_stiff_f = 2000; %前轮胎在设计状态下的侧偏刚度，N/deg
tire_corner_stiff_r = 2000; %后轮胎在设计状态下的侧偏刚度，N/deg
tire_incline_stiff_f= 134  ; %前轮胎在设计状态下的侧倾刚度，N/deg
tire_incline_stiff_r= 134  ; %后轮胎在设计状态下的侧倾刚度，N/deg


%侧向力转向贡献
fy_steer_nr_f = fy_steer_f + align_comp_f*((-nr_f)/1000); %考虑轮胎拖距的侧向力转向系数，deg/N
fy_steer_nr_r = fy_steer_r + align_comp_r*((-nr_r)/1000);
fy_toe_comp_f = fy_steer_nr_f*G_f*9.807; %前轴KC特性侧偏柔度，deg/g
fy_toe_comp_r = fy_steer_nr_r*G_r*9.807; %后轴KC特性侧偏柔度，deg/g
%侧倾转向贡献
roll_str_comp_f = roll_steer_f * total_roll_grad_t; %前悬架侧倾转向柔度
roll_str_comp_r = roll_steer_r * total_roll_grad_t; %后悬架侧倾转向柔度
%轮胎导入的贡献
tire_comp_f = -9.807*G_f/(tire_corner_stiff_f*2); %前轮胎导入的侧偏柔度
tire_comp_r = -9.807*G_f/(tire_corner_stiff_r*2); %后轮胎导入的侧偏柔度
%驱动力导入的侧偏柔度，车辆均速行驶，空气阻力系数为0.3
Fw = 0.5*0.3*area_aero*1.206*Uc^2; %空气阻力，N
fx_steer_angle_f = fx_steer_f*Fw*dirve_f;  %前轮驱动力导入的前轮前束角，deg
fx_steer_angle_r = fx_steer_r*Fw*(1-dirve_f);  %后轮驱动力导入的后轮前束角，deg
%侧向外倾的贡献
roll_camber_comp_f=tire_incline_stiff_f*roll_camber_f*total_roll_grad_t/tire_corner_stiff_f;    %前轮侧倾外倾的前轮前束角，deg
roll_camber_comp_r=tire_incline_stiff_r*roll_camber_r*total_roll_grad_t/tire_corner_stiff_r;    %后轮侧倾外倾的前轮前束角，deg
%侧向力外倾的贡献
fy_camber_comp_f=tire_incline_stiff_f*fy_camber_f*G_f/2*9.8/tire_corner_stiff_f;     %前轮侧向力外倾的前轮前束角，deg
fy_camber_comp_r=tire_incline_stiff_r*fy_camber_r*G_r/2*9.8/tire_corner_stiff_r;     %后轮侧向力外倾的前轮前束角，deg

%
comp_f = fy_toe_comp_f + roll_str_comp_f + tire_comp_f + fx_steer_angle_f + toe_f + roll_camber_comp_f + fy_camber_comp_f; %前悬架侧偏柔度，deg/g
comp_r = fy_toe_comp_r + roll_str_comp_r + tire_comp_r + fx_steer_angle_r + toe_r + roll_camber_comp_r + fy_camber_comp_r; %后悬架侧偏柔度，deg/g
us_t = -(comp_f - comp_r); %不足转向度