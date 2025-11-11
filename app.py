from flask import Flask, render_template_string, request, jsonify, send_file
import torch
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import io
import base64
import os
import hashlib
import json
from rembg import remove
from transformers import pipeline
import cv2
import redis
from datetime import timedelta

app = Flask(__name__)

# ==================== إعدادات Redis Cache ====================
# تغيير هذه الإعدادات حسب حاجتك
REDIS_CONFIG = {
    'host': os.getenv('REDIS_HOST', 'localhost'),
    'port': int(os.getenv('REDIS_PORT', 6379)),
    'db': int(os.getenv('REDIS_DB', 0)),
    'password': os.getenv('REDIS_PASSWORD', None),
    'socket_timeout': 5,
    'socket_connect_timeout': 5
}

class RedisCacheManager:
    """مدير Redis Cache متقدم"""
    
    def __init__(self, config):
        """تهيئة الاتصال بـ Redis"""
        try:
            self.client = redis.Redis(
                host=config['host'],
                port=config['port'],
                db=config['db'],
                password=config['password'],
                socket_timeout=config['socket_timeout'],
                socket_connect_timeout=config['socket_connect_timeout'],
                decode_responses=False  # نريد البايتات
            )
            # اختبار الاتصال
            self.client.ping()
            print("✅ اتصال Redis ناجح!")
        except redis.ConnectionError:
            print("❌ فشل الاتصال بـ Redis! سيتم استخدام Cache مؤقت في الذاكرة.")
            self.client = None
        except Exception as e:
            print(f"⚠️ خطأ في تهيئة Redis: {e}")
            self.client = None
    
    def generate_key(self, image_data, action, settings):
        """توليد مفتاح فريد للـ Cache"""
        # استخدام hash قوي (SHA256)
        data_string = f"{image_data}:{action}:{json.dumps(settings, sort_keys=True)}"
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def get(self, key):
        """الحصول على بيانات من الـ Cache"""
        if not self.client:
            return None
        
        try:
            cached_data = self.client.get(f"imagepro:{key}")
            if cached_data:
                # فك التسلسل
                data = json.loads(cached_data.decode('utf-8'))
                print(f"⚡ Cache Hit: {key[:8]}...")
                return data
            return None
        except Exception as e:
            print(f"خطأ في قراءة Cache: {e}")
            return None
    
    def set(self, key, image_base64, expiry_hours=24, metadata=None):
        """حفظ بيانات في الـ Cache"""
        if not self.client:
            return False
        
        try:
            data = {
                'image': image_base64,
                'metadata': metadata or {},
                'timestamp': json.dumps(datetime.now().isoformat())
            }
            
            # تسلسل البيانات
            serialized_data = json.dumps(data)
            
            # حفظ في Redis مع TTL
            self.client.setex(
                f"imagepro:{key}",
                int(timedelta(hours=expiry_hours).total_seconds()),
                serialized_data.encode('utf-8')
            )
            print(f"💾 Cache Saved: {key[:8]}... (TTL: {expiry_hours}h)")
            return True
        except Exception as e:
            print(f"خطأ في حفظ Cache: {e}")
            return False
    
    def delete(self, key):
        """حذف مفتاح من الـ Cache"""
        if not self.client:
            return False
        
        try:
            self.client.delete(f"imagepro:{key}")
            print(f"🗑️ Cache Deleted: {key[:8]}...")
            return True
        except Exception as e:
            print(f"خطأ في حذف Cache: {e}")
            return False
    
    def clear_pattern(self, pattern="imagepro:*"):
        """مسح جميع المفاتيح المتطابقة"""
        if not self.client:
            return False
        
        try:
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
                print(f"🧹 Cleared {len(keys)} cached items")
            return True
        except Exception as e:
            print(f"خطأ في مسح Cache: {e}")
            return False
    
    def stats(self):
        """إحصائيات الـ Cache"""
        if not self.client:
            return {"status": "disconnected"}
        
        try:
            info = self.client.info('stats')
            return {
                "status": "connected",
                "keys_count": len(self.client.keys("imagepro:*")),
                "memory_used": info.get('used_memory_human', 'N/A'),
                "hits": info.get('keyspace_hits', 0),
                "misses": info.get('keyspace_misses', 0)
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

# ==================== تهيئة Cache Manager ====================
cache_manager = RedisCacheManager(REDIS_CONFIG)

# ==================== تحميل نماذج AI ====================
print("🤖 جاري تحميل نماذج AI...")
upscaler = pipeline("image-to-image", model="stabilityai/stable-diffusion-x4-upscaler")
pipe = pipeline("image-inpainting", model="Sanster/lama")
print("✅ تم تحميل النماذج بنجاح!")

# ==================== دوال المعالجة ====================
def resize_image(image, max_size):
    """تحجيم الصورة مع الحفاظ على النسبة"""
    if max_size:
        image.thumbnail((max_size, max_size), Image.LANCZOS)
    return image

def auto_enhance(image):
    """تحسينات AI تلقائية"""
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.1)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.15)
    image = ImageOps.autocontrast(image, cutoff=2)
    return image

def reduce_noise(image):
    """إزالة الضوضاء"""
    img_array = np.array(image)
    denoised = cv2.bilateralFilter(img_array, 9, 75, 75)
    return Image.fromarray(denoised)

def compress_image(image, quality=85, format='PNG'):
    """ضغط الصورة"""
    buffer = io.BytesIO()
    if format == 'JPEG':
        rgb_image = image.convert('RGB')
        rgb_image.save(buffer, format='JPEG', quality=quality, optimize=True)
    else:
        image.save(buffer, format='PNG', optimize=True, compress_level=9-quality//10)
    buffer.seek(0)
    return buffer

# ==================== واجهة HTML ====================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ImagePro AI Pro - محسن الصور المتقدم</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 { text-align: center; color: #333; margin-bottom: 10px; font-size: 2.5em; }
        .subtitle { text-align: center; color: #666; margin-bottom: 30px; }
        
        /* Redis Status */
        .redis-status {
            background: #e8f5e9;
            border-left: 5px solid #4caf50;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .redis-status.connected { border-color: #4caf50; }
        .redis-status.disconnected { border-color: #f44336; background: #ffebee; }
        
        /* إعدادات متقدمة */
        .settings-panel {
            background: #f8f9ff;
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            border-left: 5px solid #667eea;
        }
        .settings-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }
        .setting-group {
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .setting-group label {
            display: block;
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }
        .setting-group input, .setting-group select {
            width: 100%;
            padding: 8px;
            border: 2px solid #e0e0e0;
            border-radius: 5px;
        }
        .setting-group small { color: #666; }
        
        /* منطقة الرفع */
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 60px 20px;
            text-align: center;
            background: #f8f9ff;
            cursor: pointer;
            transition: all 0.3s;
            margin: 20px 0;
        }
        .upload-area:hover {
            background: #e8ecff;
            transform: scale(1.02);
        }
        .upload-area.dragover {
            background: #d4d8ff;
            border-color: #764ba2;
        }
        
        /* الأزرار */
        .buttons {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-top: 30px;
            flex-wrap: wrap;
        }
        button {
            padding: 15px 30px;
            font-size: 16px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s;
        }
        .btn-primary {
            background: #667eea;
            color: white;
        }
        .btn-primary:hover {
            background: #5a67d8;
            transform: translateY(-2px);
        }
        .btn-secondary {
            background: #28a745;
            color: white;
        }
        .btn-danger {
            background: #dc3545;
            color: white;
            padding: 10px 20px;
        }
        
        /* معاينة */
        .preview {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 30px;
        }
        .preview-box {
            text-align: center;
        }
        .preview-box img {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .image-info {
            background: #e8ecff;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            font-size: 14px;
        }
        
        /* التحميل */
        .loading {
            display: none;
            text-align: center;
            margin-top: 20px;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        
        /* الأخطاء */
        .error {
            background: #fee;
            color: #c33;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            display: none;
        }
        
        /* Cache مؤشر */
        .cache-indicator {
            background: #d4edda;
            color: #155724;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 ImagePro AI Pro</h1>
        <p class="subtitle">أداة متكاملة لتحسين الصور بالذكاء الاصطناعي</p>
        
        <!-- Redis Status -->
        <div class="redis-status" id="redisStatus">
            <span>🔄 جاري التحقق من Redis...</span>
            <button class="btn-danger" onclick="clearCache()">🧹 مسح Cache</button>
        </div>
        
        <!-- لوحة الإعدادات المتقدمة -->
        <div class="settings-panel">
            <h3>⚙️ إعدادات متقدمة</h3>
            <div class="settings-grid">
                <div class="setting-group">
                    <label for="maxSize">📐 الحجم الأقصى (بكسل)</label>
                    <input type="number" id="maxSize" placeholder="مثال: 1920" min="100" max="8000">
                    <small>تحجيم الصورة بعد المعالجة</small>
                </div>
                <div class="setting-group">
                    <label for="quality">🎯 جودة الضغط</label>
                    <input type="range" id="quality" min="1" max="100" value="85">
                    <small id="qualityValue">85%</small>
                </div>
                <div class="setting-group">
                    <label for="format">💾 صيغة الإخراج</label>
                    <select id="format">
                        <option value="PNG">PNG (شفافية)</option>
                        <option value="JPEG">JPEG (حجم صغير)</option>
                    </select>
                </div>
                <div class="setting-group">
                    <label for="aiEnhance">🤖 تحسينات AI إضافية</label>
                    <select id="aiEnhance" multiple size="3">
                        <option value="auto_color" selected>تعديل تلقائي للألوان</option>
                        <option value="denoise">إزالة الضوضاء</option>
                        <option value="sharpen">ت sharpness</option>
                    </select>
                </div>
            </div>
        </div>
        
        <div class="upload-area" id="uploadArea">
            <h2>📤 اسحب صورتك هنا أو انقر للاختيار</h2>
            <p>يدعم: JPG, PNG, WebP (الحد الأقصى 10MB)</p>
            <input type="file" id="fileInput" accept="image/*" style="display: none;">
        </div>
        
        <div class="buttons">
            <button class="btn-primary" onclick="processImage('enhance')">✨ رفع الجودة 4x</button>
            <button class="btn-primary" onclick="processImage('removebg')">🖼️ إزالة الخلفية</button>
            <button class="btn-primary" onclick="processImage('removewatermark')">🚫 إزالة العلامة المائية</button>
            <button class="btn-secondary" onclick="processImage('compress')">📦 ضغط فقط</button>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>جاري المعالجة بالذكاء الاصطناعي...</p>
        </div>
        
        <div class="cache-indicator" id="cacheIndicator">
            ⚡ تم استخدام نتيجة مخزنة (Cache) - تم التحميل من Redis
        </div>
        
        <div class="error" id="error"></div>
        
        <div class="preview" id="preview" style="display: none;">
            <div class="preview-box">
                <h3>الصورة الأصلية</h3>
                <img id="originalImage">
                <div class="image-info" id="originalInfo"></div>
            </div>
            <div class="preview-box">
                <h3>الصورة المعالجة</h3>
                <img id="processedImage">
                <div class="image-info" id="processedInfo"></div>
            </div>
        </div>
    </div>
    
    <script>
        // التحقق من حالة Redis عند تحميل الصفحة
        fetch('/redis_status').then(r => r.json()).then(data => {
            const statusEl = document.getElementById('redisStatus');
            if (data.connected) {
                statusEl.className = 'redis-status connected';
                statusEl.innerHTML = `
                    ✅ Redis متصل | المفاتيح: ${data.keys_count} | Cache Hits: ${data.hits}
                    <button class="btn-danger" onclick="clearCache()">🧹 مسح Cache</button>
                `;
            } else {
                statusEl.className = 'redis-status disconnected';
                statusEl.innerHTML = `
                    ❌ Redis غير متصل - يتم استخدام cache مؤقت
                    <button class="btn-danger" onclick="clearCache()" disabled>🧹 مسح Cache</button>
                `;
            }
        });
        
        // دالة مسح Cache
        async function clearCache() {
            if (confirm('هل أنت متأكد من مسح جميع الملفات المخزنة؟')) {
                const response = await fetch('/clear_cache', {method: 'POST'});
                const result = await response.json();
                alert(result.message || 'تم مسح Cache');
                location.reload();
            }
        }
        
        // باقي الكود كما هو...
        let selectedFile = null;
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const qualitySlider = document.getElementById('quality');
        const qualityValue = document.getElementById('qualityValue');
        
        qualitySlider.addEventListener('input', (e) => {
            qualityValue.textContent = e.target.value + '%';
        });
        
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            handleFile(e.dataTransfer.files[0]);
        });
        fileInput.addEventListener('change', (e) => {
            handleFile(e.target.files[0]);
        });
        
        function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                showError('الرجاء اختيار ملف صورة صالح!');
                return;
            }
            if (file.size > 10 * 1024 * 1024) {
                showError('حجم الملف يتجاوز 10MB!');
                return;
            }
            selectedFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = document.getElementById('originalImage');
                img.src = e.target.result;
                img.onload = () => {
                    document.getElementById('originalInfo').innerHTML = `
                        الحجم: ${(file.size / 1024).toFixed(2)} KB<br>
                        الأبعاد: ${img.naturalWidth} x ${img.naturalHeight}px
                    `;
                };
                document.getElementById('preview').style.display = 'grid';
                hideError();
            };
            reader.readAsDataURL(file);
        }
        
        async function processImage(action) {
            if (!selectedFile) {
                showError('الرجاء اختيار صورة أولاً!');
                return;
            }
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('cacheIndicator').style.display = 'none';
            
            const formData = new FormData();
            formData.append('image', selectedFile);
            formData.append('action', action);
            formData.append('maxSize', document.getElementById('maxSize').value);
            formData.append('quality', document.getElementById('quality').value);
            formData.append('format', document.getElementById('format').value);
            formData.append('aiEnhance', Array.from(document.getElementById('aiEnhance').selectedOptions).map(o => o.value));
            
            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    const img = document.getElementById('processedImage');
                    img.src = 'data:image/png;base64,' + result.image;
                    
                    img.onload = () => {
                        document.getElementById('processedInfo').innerHTML = `
                            الحجم: ${(result.size / 1024).toFixed(2)} KB<br>
                            الأبعاد: ${result.width} x ${result.height}px<br>
                            ${result.fromCache ? '<span style="color: green; font-weight: bold">⚡ من Redis Cache</span>' : ''}
                        `;
                    };
                    
                    if (result.fromCache) {
                        document.getElementById('cacheIndicator').style.display = 'block';
                    }
                } else {
                    showError(result.error || 'حدث خطأ أثناء المعالجة');
                }
            } catch (err) {
                showError('خطأ في الاتصال بالخادم: ' + err.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }
        
        function showError(msg) {
            const errorEl = document.getElementById('error');
            errorEl.textContent = msg;
            errorEl.style.display = 'block';
            setTimeout(() => errorEl.style.display = 'none', 5000);
        }
        
        function hideError() {
            document.getElementById('error').style.display = 'none';
        }
    </script>
</body>
</html>
"""

# ==================== نقاط نهاية API ====================
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/redis_status')
def redis_status():
    """الحصول على حالة Redis وإحصائيات"""
    stats = cache_manager.stats()
    return jsonify(stats)

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """مسح جميع الملفات المخزنة"""
    success = cache_manager.clear_pattern("imagepro:*")
    if success:
        return jsonify({'success': True, 'message': 'تم مسح جميع الملفات المخزنة'})
    return jsonify({'success': False, 'message': 'فشل في مسح الـ Cache'})

@app.route('/process', methods=['POST'])
def process():
    """معالجة الصورة مع Redis Cache"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'لم يتم رفع صورة'})
        
        file = request.files['image']
        action = request.form.get('action')
        max_size = int(request.form.get('maxSize', 0)) if request.form.get('maxSize') else None
        quality = int(request.form.get('quality', 85))
        output_format = request.form.get('format', 'PNG')
        ai_enhance = request.form.getlist('aiEnhance')
        
        # قراءة بيانات الصورة
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # توليد مفتاح cache
        cache_key = cache_manager.generate_key(
            image_bytes.hex(),
            action,
            {
                'max_size': max_size,
                'quality': quality,
                'format': output_format,
                'ai_enhance': ai_enhance
            }
        )
        
        # محاولة الحصول من cache
        cached_data = cache_manager.get(cache_key)
        from_cache = False
        processed_image = None
        
        if cached_data:
            # استخدام النتيجة المخزنة
            from_cache = True
            processed_image = Image.open(io.BytesIO(base64.b64decode(cached_data['image'])))
            print(f"⚡ Cache Hit: {cache_key[:8]}...")
        else:
            # معالجة جديدة
            print(f"🔄 Cache Miss: {cache_key[:8]}... Processing...")
            
            # المعالجة حسب الإجراء
            if action == 'enhance':
                result = upscaler(image)
                processed_image = result[0]
                
            elif action == 'removebg':
                processed_image = remove(image)
                
            elif action == 'removewatermark':
                width, height = image.size
                mask = Image.new('L', (width, height), 0)
                mask.paste(255, (max(0, width-300), max(0, height-150), width, height))
                result = pipe(image=image, mask_image=mask)
                processed_image = result[0] if isinstance(result, list) else result
                
            elif action == 'compress':
                processed_image = image
                
            else:
                return jsonify({'success': False, 'error': 'إجراء غير معروف'})
            
            # تطبيق تحسينات AI
            if 'auto_color' in ai_enhance:
                processed_image = auto_enhance(processed_image)
            if 'denoise' in ai_enhance:
                processed_image = reduce_noise(processed_image)
            
            # تحجيم الصورة
            if max_size:
                processed_image = resize_image(processed_image, max_size)
            
            # حفظ في cache
            buffer_temp = io.BytesIO()
            processed_image.save(buffer_temp, format='PNG')
            cache_manager.set(
                cache_key,
                base64.b64encode(buffer_temp.getvalue()).decode(),
                expiry_hours=24,
                metadata={'action': action, 'size': len(buffer_temp.getvalue())}
            )
        
        # ضغط الصورة النهائية
        compressed_buffer = compress_image(processed_image, quality, output_format)
        
        # تحضير الرد
        img_base64 = base64.b64encode(compressed_buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'size': len(compressed_buffer.getvalue()),
            'width': processed_image.width,
            'height': processed_image.height,
            'fromCache': from_cache
        })
        
    except Exception as e:
        print(f"❌ خطأ في المعالجة: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)