#include <iostream>

using namespace std;

class Complex{
public:
    Complex(){}
    Complex(float real, float image){
        //TO DO
        _real=real;
        _image=image;
    }
    const Complex operator+(const Complex& k){
        // TO DO
        return Complex(_real+k._real,_image+k._image);
    }
    const Complex operator*(const Complex& k){
        // TO DO
        return Complex(_real*k._real-_image*k._image,_real*k._image+_image*k._real);
    }
    float getReal() const{
        // TO DO
        return _real;

    }
    float getImage() const{
        // TO DO
        return _image;
    }
private:
    float _real,_image;
};

ostream& operator<<(ostream& out,const Complex& k){
    float real, image;
    real = k.getReal();
    image = k.getImage();
    if(image >= 0)out<<real<<" + "<<image<<"i";
      else out<<real<<" - "<<-image<<"i";
    return out;
}

int main(){
    float real1, image1;
    float real2, image2;
    float real3, image3;
    while(cin >> real1 >> image1 >> real2 >> image2){
        Complex myComplex1(real1, image1);
        Complex myComplex2(real2, image2);
        cout<< myComplex1 + myComplex2<< endl;
        cout<< myComplex1 * myComplex2<< endl;

    }
    return 0;
}
