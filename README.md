# re_data

این تابع تاریخچه قیمتی را در قالب یک دیتافریم پانداس دریافت و منابع مناسب برای داده کاوی را استخراج میکند

Load:
تعداد کندل های گذشته که به عنوان فیچر های دیتا در نظر میگیرد

nn:
قیمت ان ان کندل آینده تقسیم بر قیمت کنونی را به عنوان برچسب هر روز در نظر میگیرد

train_size:
اندازه قسمتی از داده را که به عنوان داده آموزشی در خروجی تحویل میدهد لازم به ذکر است به دلیل مرتب بودن داده های مالی ترین و تست رندم انتخاب نشده و داده های تست همگی بعد از ترین انتخاب شده اند



روش کار به این شکل است که داده های لود روز گذشته را به ترتیب کنار هم قرار میدهد و تقسیم بر قیمت اوپن امروز میکند تا تاثیر خود قیمت را از بین ببرد و فقط نسبت های قیمتی را حفظ کند

توجه داشته باشید که دیتای ورودی باید با چهار ستون اولین قیمت و بیشترین و کم ترین و قیمت پایانی شروع شده باشد و ستون های دیگر بعد از این ها باشند
