import streamlit as st  # استيراد مكتبة Streamlit لبناء واجهة المستخدم
import tensorflow as tf  # استيراد TensorFlow لاستخدام النماذج المدربة
from PIL import Image  # استيراد PIL لتحميل ومعالجة الصور
import numpy as np  # استيراد NumPy للعمل مع المصفوفات
import matplotlib.pyplot as plt  # استيراد مكتبة matplotlib للرسم البياني


# تحميل النماذج المدربة (استبدل بالمسار الصحيح للنماذج)
model_cnn1 = tf.keras.models.load_model("C:/Users/HUAWI/Multipage/Models/Model1.h5")
model_cnn2 = tf.keras.models.load_model("C:/Users/HUAWI/Multipage/Models/Model2.h5")
model_cnn3 = tf.keras.models.load_model("C:/Users/HUAWI/Multipage/Models/Model3.h5")

# قائمة الأسماء المحتملة للفئات (تأكد من أنها تتوافق مع مخرجات النموذج)
class_names = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 
               'elephant', 'horse', 'sheep', 'spider', 'squirrel']

# دالة التنبؤ
def predict(image, model):
    # تحويل الصورة إلى مصفوفة وتقسيمها على 255 لتطبيع القيم بين 0 و 1
    img_array = np.array(image) / 255.0
    # تغيير حجم الصورة إلى الأبعاد المطلوبة من قبل النموذج
    img_array = tf.image.resize(img_array, [model.input_shape[1], model.input_shape[2]])
    # إضافة بعد جديد ليصبح لدينا حجم الدفعة (batch size)
    img_array = np.expand_dims(img_array, axis=0)
    
    # الحصول على التنبؤات من النموذج
    prediction = model.predict(img_array)
    # استخراج الفئة ذات الاحتمال الأعلى
    class_id = np.argmax(prediction)
    return class_id, prediction  # إرجاع الفئة والاحتمالات

# دالة الصفحة الرئيسية
def run_homepage():
    # عنوان الصفحة
    st.title("Welcome to the Image Classification App using AI")
    
    # وصف موجز للتطبيق
   
    st.write("**Neural Vision:**")
    st.write("Ranwah sadik")
    st.write("Razan alkhamisi")
    st.write("Wejdan alharthi")
    st.write(" ")
    st.write("In this app, our team developed 3 models for image classification.")
    st.write("Here are the available models:")
    st.write("- **CNN Model 1 : MobileNet**")
    st.write("The **MobileNet model** achieves high accuracy (95% on training, 94% on validation and test sets), showcasing its effective learning and generalization capabilities through transfer learning.")
    st.write("- **CNN Model 2 : ResNet**")
    st.write("We use the **ResNet model** in transfer learning to leverage pre-trained features, improving our model’s accuracy by adapting it to a new task with fewer data.")
    st.write("- **CNN Model 3 : Our First CNN Model**")
    st.write("The **CNN first model** achieves 67% accuracy")
    st.write(" ")
    st.write("**Choose a model from the sidebar to start**")
    
    # واجهة المستخدم لاختيار النموذج من الشريط الجانبي
    # واجهة المستخدم لاختيار النموذج من الشريط الجانبي
    page = st.sidebar.selectbox("Choose a model", ["Mobilenet Model", "ResNet Model", "CNN Model"])


    # رفع الصورة من المستخدم
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)  # فتح الصورة
        st.image(image, caption="Uploaded Image", use_column_width=True)  # عرض الصورة المرفوعة
        
        # التنبؤ باستخدام النموذج المحدد
        if page == "Mobilenet Model":
            st.title("CNN Model 1")
            class_id, probabilities = predict(image, model_cnn1)  # التنبؤ باستخدام النموذج الأول
            st.write(f"Predicted class: {class_names[class_id]}")  # عرض الفئة المتوقعة
            

            # رسم البياني
            fig, ax = plt.subplots()
            ax.bar(class_names, probabilities[0])  # رسم الأعمدة
            ax.set_xlabel('Class')  # تسميات المحور الأفقي
            ax.set_ylabel('Probability')  # تسميات المحور الرأسي
            ax.set_title('Class Probabilities')  # عنوان الرسم البياني
            plt.xticks(rotation=90)  # تدوير التسميات لتكون أكثر وضوحًا
            st.pyplot(fig)  # عرض الرسم البياني في Streamlit

        elif page == "ResNet Model":
            st.title("CNN Model 2")
            class_id, probabilities = predict(image, model_cnn2)  # التنبؤ باستخدام النموذج الثاني
            st.write(f"Predicted class: {class_names[class_id]}")  # عرض الفئة المتوقعة
            

            # رسم البياني
            fig, ax = plt.subplots()
            ax.bar(class_names, probabilities[0])  # رسم الأعمدة
            ax.set_xlabel('Class')  # تسميات المحور الأفقي
            ax.set_ylabel('Probability')  # تسميات المحور الرأسي
            ax.set_title('Class Probabilities')  # عنوان الرسم البياني
            plt.xticks(rotation=90)  # تدوير التسميات لتكون أكثر وضوحًا
            st.pyplot(fig)  # عرض الرسم البياني في Streamlit


        elif page == "CNN Model":
           st.title("CNN Model 3")
           class_id, probabilities = predict(image, model_cnn3)  # التنبؤ باستخدام النموذج الثالث
           st.write(f"Predicted class: {class_names[class_id]}")  # عرض الفئة المتوقعة
           


            # رسم البياني
           fig, ax = plt.subplots()
           ax.bar(class_names, probabilities[0])  # رسم الأعمدة
           ax.set_xlabel('Class')  # تسميات المحور الأفقي
           ax.set_ylabel('Probability')  # تسميات المحور الرأسي
           ax.set_title('Class Probabilities')  # عنوان الرسم البياني
           plt.xticks(rotation=90)  # تدوير التسميات لتكون أكثر وضوحًا
           st.pyplot(fig)  # عرض الرسم البياني في Streamlit

# تشغيل التطبيق
if __name__ == "__main__":
    run_homepage()  # استدعاء دالة الصفحة الرئيسية لتشغيل التطبيق
