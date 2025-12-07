"""
⚠️ AKADEMİK SİSTEM ANALİZ ÇERÇEVESİ - Savunma Amaçlı Araştırma Aracı
Bu kod, metin benzerliği ve AI dedeksiyon sistemlerinin çalışma prensiplerini
anlamak için tasarlanmıştır. Akademik sahtekarlık amaçlı kullanımı şiddetle tavsiye edilmez
ve ciddi yaptırımlarla sonuçlanabilir.
"""

import nltk
import random
import requests
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass
import hashlib

# ==================== ÇEKİRDEK DÖNÜŞÜM SİSTEMİ ====================

@dataclass
class EvasionResult:
    """Dönüşüm işleminin sonucunu tutar."""
    original_text: str
    transformed_text: str
    techniques_used: List[str]
    risk_score: float  # 0-1 arası, 1 = yüksek risk
    detected_patterns: List[str]

class AdvancedTextTransformer:
    """
    Rehberdeki Part 1 & 2 tekniklerini uygular.
    AMAÇ: Sistemlerin zafiyetlerini anlamak, pratik bypass için değil.
    """
    
    def __init__(self, aggression_level: float = 0.7):
        """
        :param aggression_level: 0-1 arası. Ne kadar yüksekse, metin o kadar çok değişir (ve doğallıktan o kadar uzaklaşır).
        """
        self.aggression = aggression_level
        self.used_techniques = []
        
        # Çoklu dil desteği için (Back-Translation)
        self.supported_langs = ['es', 'fr', 'de', 'it', 'ru']
        
        # Kişiselleştirme havuzu
        self.personal_templates = [
            "Bunu şahsen deneyimledim, {example}.",
            "Okuduğum bir makalede, {reference} belirtiliyordu.",
            "Kendi projemde gördüm ki, {insight}.",
            "Meslektaşlarımla yaptığımız tartışmada, {opinion} sonucuna vardık.",
            "Geçen sene katıldığım {conference} konferansında bu konu detaylıca ele alınmıştı."
        ]
    
    def _calculate_risk(self) -> float:
        """Aşırı dönüşümün doğallığı bozma ve tespit edilme riskini hesaplar."""
        base_risk = self.aggression * 0.6
        technique_penalty = len(self.used_techniques) * 0.05
        return min(0.95, base_risk + technique_penalty)
    
    def apply_deep_paraphrase(self, text: str) -> str:
        """1.1 Deep and Thorough Paraphrasing"""
        # Burada basit bir synonym değişimi yapılıyor.
        # GERÇEK bir uygulama, BERT gibi modellerle anlamsal paraphrasing gerektirir.
        synonym_map = {
            'cause': ['trigger', 'lead to', 'result in', 'bring about'],
            'important': ['crucial', 'vital', 'significant', 'essential'],
            'method': ['approach', 'technique', 'procedure', 'strategy'],
            'analysis': ['examination', 'study', 'investigation', 'evaluation'],
            'result': ['outcome', 'finding', 'conclusion', 'product']
        }
        
        words = nltk.word_tokenize(text)
        new_words = []
        for word in words:
            lower_word = word.lower()
            if lower_word in synonym_map and random.random() < 0.3:
                new_word = random.choice(synonym_map[lower_word])
                # Orijinal büyük/küçük harf durumunu koru
                if word[0].isupper():
                    new_word = new_word.capitalize()
                new_words.append(new_word)
            else:
                new_words.append(word)
        
        self.used_techniques.append("deep_paraphrasing")
        return ' '.join(new_words)
    
    def apply_sentence_manipulation(self, text: str) -> str:
        """1.6 Sentence Manipulation"""
        sentences = nltk.sent_tokenize(text)
        if len(sentences) < 2:
            return text
        
        manipulated = []
        for sent in sentences:
            words = nltk.word_tokenize(sent)
            # Rastgele cümleleri böl veya birleştir
            if len(words) > 15 and random.random() < 0.4:
                # Uzun cümleyi böl
                mid = len(words) // 2
                part1 = ' '.join(words[:mid]) + '.'
                part2 = ' '.join(words[mid:]) + '.'
                manipulated.extend([part1, part2])
            elif len(words) < 5 and len(manipulated) > 0 and random.random() < 0.3:
                # Kısa cümleyi önceki cümleye ekle
                last_sent = manipulated.pop().rstrip('.')
                combined = f"{last_sent}, and {' '.join(words).lower()}."
                manipulated.append(combined)
            else:
                manipulated.append(sent)
        
        self.used_techniques.append("sentence_manipulation")
        return ' '.join(manipulated)
    
    def back_translation_round(self, text: str, lang_pair: Tuple[str, str]) -> str:
        """1.5 Translation and Back-Translation (Simülasyon)"""
        # GERÇEK uygulama Google Translate API veya benzeri gerektirir.
        # Bu basit bir simülasyondur.
        src, tgt = lang_pair
        
        # Simüle edilmiş çeviri hataları/kelime değişimleri
        simulation_map = {
            'the': ['a', 'this', 'that'],
            'is': ['was', 'becomes', 'represents'],
            'of': ['from', 'in', 'about'],
            'and': ['as well as', 'plus', 'along with'],
            'in': ['within', 'inside', 'during']
        }
        
        words = text.split()
        translated = []
        for word in words:
            clean_word = word.lower().strip('.,!?;:')
            if clean_word in simulation_map and random.random() < 0.2:
                translated.append(random.choice(simulation_map[clean_word]))
            else:
                translated.append(word)
        
        # Dil çiftine özgü "yanlış çeviri" simülasyonu
        if tgt == 'de':  # Almanca
            if random.random() < 0.1:
                translated.append("Actually,")
        elif tgt == 'fr':  # Fransızca
            if random.random() < 0.1:
                translated.insert(0, "Well,")
        
        self.used_techniques.append(f"back_translation_{src}_{tgt}")
        return ' '.join(translated)
    
    def add_personal_touch(self, text: str) -> str:
        """2.2 Humanizing Techniques"""
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return text
        
        # Rastgele bir yere kişisel anekdot ekle
        if random.random() < 0.4:
            insert_point = random.randint(0, len(sentences)-1)
            template = random.choice(self.personal_templates)
            
            # Template'i doldur
            filled_template = template.format(
                example="örneğin benzer bir sorunla karşılaşmıştım",
                reference="bu konunun önemi vurgulanıyor",
                insight="bu yaklaşım her zaman işe yaramıyor",
                opinion="daha kapsamlı bir analiz gerekli",
                conference="IEEE"
            )
            
            sentences.insert(insert_point, filled_template)
        
        # Kasıtlı küçük "hatalar" ekle (çok dikkatli!)
        if random.random() < 0.1 and self.aggression > 0.5:
            # Sadece yüksek aggression seviyesinde ve nadiren
            error_sentence = random.randint(0, len(sentences)-1)
            words = sentences[error_sentence].split()
            if len(words) > 3:
                # "its" vs "it's" karışıklığı simülasyonu
                if "its" in words or "it's" in words:
                    sentences[error_sentence] = sentences[error_sentence].replace("its", "[its/it's]")
        
        self.used_techniques.append("personalization")
        return ' '.join(sentences)
    
    def vary_sentence_structure(self, text: str) -> str:
        """2.2 Vary Sentence Structure Drastically"""
        sentences = nltk.sent_tokenize(text)
        if len(sentences) < 3:
            return text
        
        varied = []
        lengths = []
        for i, sent in enumerate(sentences):
            words = sent.split()
            lengths.append(len(words))
            
            # Cümle başlangıçlarını çeşitlendir
            starters = ['However, ', 'Furthermore, ', 'In contrast, ', 
                       'Specifically, ', 'Typically, ', 'As a result, ']
            
            if i > 0 and random.random() < 0.3:
                sent = random.choice(starters) + sent[0].lower() + sent[1:]
            
            varied.append(sent)
        
        # Burstiness kontrolü: Çok düzenliyse, kısa bir cümle ekle
        if len(set(lengths)) < 3 and random.random() < 0.5:
            short_sentences = ["This is key.", "Consider this.", "Note this point."]
            insert_pos = random.randint(1, len(varied)-1)
            varied.insert(insert_pos, random.choice(short_sentences))
        
        self.used_techniques.append("sentence_variation")
        return ' '.join(varied)
    
    def execute_full_pipeline(self, text: str) -> EvasionResult:
        """
        Rehberdeki Part 3 kombine stratejisini uygular.
        """
        original_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        print(f"[*] Orijinal metin işleniyor (hash: {original_hash})...")
        
        current_text = text
        stages = []
        
        # 1. Temel Paraphrasing
        current_text = self.apply_deep_paraphrase(current_text)
        stages.append("initial_paraphrase")
        
        # 2. Cümle Manipülasyonu
        current_text = self.apply_sentence_manipulation(current_text)
        stages.append("sentence_restructure")
        
        # 3. Back-Translation (1-2 tur)
        for i in range(random.randint(1, 2)):
            lang_pair = random.choice([('en', 'fr'), ('en', 'de'), ('en', 'es')])
            current_text = self.back_translation_round(current_text, lang_pair)
            stages.append(f"translation_round_{i+1}")
        
        # 4. İnsanlaştırma
        current_text = self.add_personal_touch(current_text)
        stages.append("humanization")
        
        # 5. Cümle Yapısı Çeşitlendirme
        current_text = self.vary_sentence_structure(current_text)
        stages.append("final_variation")
        
        # Risk Analizi
        risk = self._calculate_risk()
        detected_patterns = []
        
        # Potansiyel olarak tespit edilebilecek kalıpları belirle
        if len(self.used_techniques) > 4:
            detected_patterns.append("excessive_transformation")
        if "back_translation" in ' '.join(self.used_techniques):
            detected_patterns.append("translation_artifacts")
        if risk > 0.7:
            detected_patterns.append("high_unnaturalness")
        
        return EvasionResult(
            original_text=text,
            transformed_text=current_text,
            techniques_used=self.used_techniques,
            risk_score=risk,
            detected_patterns=detected_patterns
        )

# ==================== DEDEKSİYON TEST ÇERÇEVESİ ====================

class DetectionTester:
    """
    2.6 Testing and Iterative Refinement bölümünü simüle eder.
    GERÇEK dedektörler yerine, basit kalıp eşleştirmesi yapar.
    """
    
    def __init__(self):
        self.ai_patterns = [
            r'\b(furthermore|moreover|however|therefore|consequently)\b',
            r'\. {2,}[A-Z]',  # İki boşlukla başlayan cümle
            r'\bin conclusion\b',
            r'\bit is important to note that\b',
            r'\bthis (suggests|indicates|implies) that\b'
        ]
        
        self.plagiarism_indicators = [
            "according to",
            "as stated by",
            "research shows",
            "studies have found",
            "it has been proven"
        ]
    
    def analyze_text(self, text: str) -> Dict:
        """Metni basit dedeksiyon kurallarına göre analiz eder."""
        import re
        
        results = {
            'ai_score': 0.0,
            'plagiarism_score': 0.0,
            'flags': [],
            'recommendations': []
        }
        
        # AI Dedeksiyonu
        ai_matches = 0
        for pattern in self.ai_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            ai_matches += len(matches)
        
        results['ai_score'] = min(1.0, ai_matches / 10)  # Normalize
        
        # İntihal İndikatörleri
        plagiarism_matches = 0
        words = text.lower().split()
        for indicator in self.plagiarism_indicators:
            if indicator in text.lower():
                plagiarism_matches += 1
        
        results['plagiarism_score'] = min(1.0, plagiarism_matches / 5)
        
        # Bayraklar
        if results['ai_score'] > 0.3:
            results['flags'].append("AI-like patterns detected")
        if results['plagiarism_score'] > 0.4:
            results['flags'].append("Potential plagiarism phrasing")
        
        if len(text.split()) < 100:
            results['flags'].append("Text may be too short for reliable analysis")
        
        # Öneriler
        if results['ai_score'] > 0.5:
            results['recommendations'].append("Add more personal anecdotes")
            results['recommendations'].append("Vary sentence length more")
        
        if results['plagiarism_score'] > 0.6:
            results['recommendations'].append("Increase original analysis")
            results['recommendations'].append("Use more unique phrasing")
        
        return results
    
    def iterative_refinement(self, transformer: AdvancedTextTransformer, 
                           original_text: str, 
                           max_iterations: int = 5,
                           target_ai_score: float = 0.2,
                           target_plagiarism_score: float = 0.3) -> Dict:
        """
        2.6 Testing and Iterative Refinement
        Belirlenen hedeflere ulaşana kadar dönüşüm uygular.
        """
        print(f"[*] Iteratif iyileştirme başlatılıyor (max {max_iterations} iterasyon)...")
        
        best_result = None
        best_score = float('inf')
        history = []
        
        for iteration in range(max_iterations):
            print(f"  Iterasyon {iteration + 1}/{max_iterations}...")
            
            # Dönüşümü uygula (her iterasyonda aggression'ı biraz düşür)
            transformer.aggression = max(0.3, transformer.aggression * 0.8)
            result = transformer.execute_full_pipeline(original_text)
            
            # Analiz et
            analysis = self.analyze_text(result.transformed_text)
            
            # Skor hesapla (ne kadar düşükse o kadar iyi)
            score = (analysis['ai_score'] * 0.6 + 
                    analysis['plagiarism_score'] * 0.4 +
                    result.risk_score * 0.3)
            
            history.append({
                'iteration': iteration + 1,
                'text': result.transformed_text[:100] + "...",
                'ai_score': analysis['ai_score'],
                'plagiarism_score': analysis['plagiarism_score'],
                'risk_score': result.risk_score,
                'total_score': score,
                'techniques_used': result.techniques_used
            })
            
            # En iyi sonucu güncelle
            if score < best_score:
                best_score = score
                best_result = {
                    'transformation_result': result,
                    'analysis': analysis,
                    'iteration': iteration + 1,
                    'total_score': score
                }
            
            # Hedeflere ulaşıldı mı?
            if (analysis['ai_score'] <= target_ai_score and 
                analysis['plagiarism_score'] <= target_plagiarism_score):
                print(f"  [+] Hedeflere {iteration + 1}. iterasyonda ulaşıldı!")
                break
        
        return {
            'best_result': best_result,
            'history': history,
            'targets_met': best_result is not None and 
                          best_result['analysis']['ai_score'] <= target_ai_score and
                          best_result['analysis']['plagiarism_score'] <= target_plagiarism_score
        }

# ==================== ETİK VE RİSK DEĞERLENDİRMESİ ====================

class EthicalAuditor:
    """
    Kullanımı izler ve etik ihlalleri raporlar.
    Bu kısım BİR PRODUKSİYON SİSTEMİNDE ZORUNLU OLMALIDIR.
    """
    
    def __init__(self):
        self.warning_levels = {
            'LOW': 'Düşük risk - Araştırma amaçlı kullanım',
            'MEDIUM': 'Orta risk - Şüpheli kullanım modelleri',
            'HIGH': 'Yüksek risk - Potansiyel akademik sahtekarlık',
            'CRITICAL': 'Kritik risk - Ticari veya kitlesel sahtekarlık'
        }
        
    def audit_usage(self, original_text: str, transformed_text: str, 
                   techniques_used: List[str], context: str = "unknown") -> Dict:
        """
        Kullanımı denetler ve risk seviyesini belirler.
        """
        # Basit heuristics (gerçek bir sistem çok daha karmaşık olmalı)
        word_reduction = 1 - (len(transformed_text.split()) / max(1, len(original_text.split())))
        
        risk_factors = []
        
        if word_reduction > 0.5:
            risk_factors.append(f"excessive_text_reduction ({word_reduction:.0%})")
        
        if len(techniques_used) > 5:
            risk_factors.append("too_many_transformation_techniques")
        
        if 'back_translation' in str(techniques_used):
            risk_factors.append("translation-based_obfuscation")
        
        # Risk seviyesini belirle
        risk_score = len(risk_factors) * 0.2 + word_reduction * 0.3
        
        if risk_score < 0.3:
            level = 'LOW'
        elif risk_score < 0.6:
            level = 'MEDIUM'
        elif risk_score < 0.8:
            level = 'HIGH'
        else:
            level = 'CRITICAL'
        
        # Akademik sahtekarlık uyarıları
        warnings = []
        if level in ['HIGH', 'CRITICAL']:
            warnings.append("⚠️ Bu kullanım akademik sahtekarlık olarak değerlendirilebilir.")
            warnings.append("⚠️ Olası sonuçlar: Ders notunun iptali, disiplin cezası, okuldan atılma.")
            warnings.append("⚠️ Akademik kurumunuzun dürüstlük politikasını kontrol edin.")
        
        return {
            'risk_level': level,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'warning': self.warning_levels[level],
            'specific_warnings': warnings,
            'recommendation': self._get_recommendation(level, context)
        }
    
    def _get_recommendation(self, risk_level: str, context: str) -> str:
        """Risk seviyesine göre öneri verir."""
        recommendations = {
            'LOW': "Kullanım araştırma amaçlı görünüyor. Metin analizi tekniklerini öğrenmeye devam edin.",
            'MEDIUM': "Dikkatli olun. Yalnızca kendi yazdığınız metinler üzerinde deney yapın.",
            'HIGH': "DURUN. Bu aktivite akademik sahtekarlık olarak sınıflandırılabilir.",
            'CRITICAL': "HEMEN DURUN. Bu yazılımı bu şekilde kullanmak etik değil ve yasa dışı olabilir."
        }
        return recommendations.get(risk_level, "Risk seviyesi değerlendirilemedi.")

# ==================== ANA KULLANIM ÖRNEĞİ ====================

def demonstrate_system():
    """Sistemin araştırma amaçlı nasıl kullanılabileceğini gösterir."""
    
    print("=" * 70)
    print("AKADEMİK SİSTEM ANALİZ ÇERÇEVESİ - DEMONSTRASYON")
    print("AMAC: Metin dedeksiyon sistemlerinin çalışma prensiplerini anlamak")
    print("=" * 70)
    print()
    
    # Örnek metin (AI tarafından üretilmiş gibi görünen)
    sample_text = """
    Artificial intelligence has transformed numerous industries through its ability to 
    analyze large datasets and identify patterns. Machine learning algorithms enable 
    systems to improve their performance over time without explicit programming. 
    Furthermore, deep learning techniques have achieved remarkable success in image 
    recognition and natural language processing. However, ethical considerations 
    regarding bias and transparency remain important challenges that require 
    continuous attention from researchers and practitioners.
    """
    
    print("[1/4] Örnek metin yükleniyor...")
    print(f"Metin: {sample_text[:150]}...")
    print(f"Uzunluk: {len(sample_text.split())} kelime")
    print()
    
    print("[2/4] Gelişmiş dönüşüm uygulanıyor...")
    transformer = AdvancedTextTransformer(aggression_level=0.6)
    result = transformer.execute_full_pipeline(sample_text)
    
    print(f"Kullanılan teknikler: {', '.join(result.techniques_used)}")
    print(f"Risk skoru: {result.risk_score:.2f}")
    if result.detected_patterns:
        print(f"Tespit edilebilecek kalıplar: {', '.join(result.detected_patterns)}")
    print()
    
    print("[3/4] Dedeksiyon analizi yapılıyor...")
    tester = DetectionTester()
    analysis = tester.analyze_text(result.transformed_text)
    
    print(f"AI skoru: {analysis['ai_score']:.2f} (hedef: <0.20)")
    print(f"İntihal skoru: {analysis['plagiarism_score']:.2f} (hedef: <0.30)")
    if analysis['flags']:
        print(f"Bayraklar: {', '.join(analysis['flags'])}")
    print()
    
    print("[4/4] Etik denetim yapılıyor...")
    auditor = EthicalAuditor()
    ethics_report = auditor.audit_usage(
        original_text=sample_text,
        transformed_text=result.transformed_text,
        techniques_used=result.techniques_used,
        context="educational_research"
    )
    
    print(f"Risk seviyesi: {ethics_report['risk_level']}")
    print(f"Uyarı: {ethics_report['warning']}")
    if ethics_report['specific_warnings']:
        for warning in ethics_report['specific_warnings']:
            print(f"  {warning}")
    print(f"Öneri: {ethics_report['recommendation']}")
    print()
    
    print("=" * 70)
    print("DÖNÜŞTÜRÜLMÜŞ METİN ÖRNEĞİ (ilk 300 karakter):")
    print("-" * 70)
    print(result.transformed_text[:300] + "...")
    print("-" * 70)
    
    # Iteratif iyileştirme örneği (isteğe bağlı)
    print()
    print("=" * 70)
    print("İTERATİF İYİLEŞTİRME DEMOSU (isteğe bağlı)")
    print("=" * 70)
    
    iterative_result = tester.iterative_refinement(
        transformer=AdvancedTextTransformer(aggression_level=0.8),
        original_text=sample_text,
        max_iterations=3,
        target_ai_score=0.2,
        target_plagiarism_score=0.3
    )
    
    if iterative_result['best_result']:
        best = iterative_result['best_result']
        print(f"En iyi sonuç {best['iteration']}. iterasyonda:")
        print(f"  Toplam skor: {best['total_score']:.3f}")
        print(f"  AI skoru: {best['analysis']['ai_score']:.2f}")
        print(f"  İntihal skoru: {best['analysis']['plagiarism_score']:.2f}")
        print(f"  Hedeflere ulaşıldı mı? {'Evet' if iterative_result['targets_met'] else 'Hayır'}")
    
    return {
        'transformer': transformer,
        'result': result,
        'analysis': analysis,
        'ethics_report': ethics_report,
        'iterative_result': iterative_result if 'iterative_result' in locals() else None
    }

# ==================== UYARI VE SÖZLEŞME ====================

def display_legal_warning():
    """Kullanıcıya yasal ve etik uyarıları gösterir."""
    warning = """
    ⚠️ ⚠️ ⚠️ ÖNEMLİ UYARI VE SÖZLEŞME ⚠️ ⚠️ ⚠️
    
    BU YAZILIM YALNIZCA ARAŞTIRMA VE EĞİTİM AMAÇLIDIR.
    
    KULLANIM KOŞULLARI:
    1. Bu araç yalnızca:
       - Akademik sistemlerin çalışma prensiplerini anlamak
       - Metin analizi tekniklerini öğrenmek
       - Savunma amaçlı güvenlik araştırması yapmak
       için kullanılabilir.
    
    2. KESİNLİKLE YAPILMASI YASAK OLANLAR:
       - Başkalarının çalışmalarını intihal etmek
       - AI ile üretilmiş metinleri özgün çalışma gibi sunmak
       - Akademik veya ticari sahtekarlık yapmak
       - Herhangi bir akademik kurumun dürüstlük politikasını ihlal etmek
    
    3. POTANSİYEL SONUÇLAR:
       - Akademik cezalar (ders notunun iptali, okuldan atılma)
       - Yasal işlem (fikri mülkiyet ihlali)
       - Mesleki itibar kaybı
       - Kalıcı akademik sicil
    
    4. SORUMLULUK REDDİ:
       Bu yazılımın yanlış kullanımından doğacak tüm sonuçlar
       TAMAMEN KULLANICIYA AİTTİR. Geliştiriciler hiçbir sorumluluk
       kabul etmez.
    
    Bu uyarıyı okudum ve anladım: [E/H]
    """
    
    print(warning)
    response = input("Cevabınız: ").strip().upper()
    
    if response != 'E':
        print("Kullanım reddedildi. Program sonlandırılıyor.")
        exit(1)
    
    print("Sözleşme kabul edildi. Araştırma moduna geçiliyor...")
    print()

# ==================== ANA PROGRAM ====================

if __name__ == "__main__":
    """
    Bu programı çalıştırmak için:
    python academic_analysis_framework.py
    
    Veya demo modu için:
    python academic_analysis_framework.py --demo
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Akademik Sistem Analiz Çerçevesi - Araştırma Aracı',
        epilog='UYARI: Yalnızca etik araştırma için kullanın.'
    )
    parser.add_argument('--demo', action='store_true', 
                       help='Demonstrasyon modunu çalıştır')
    parser.add_argument('--text', type=str, 
                       help='Analiz edilecek metin (tırnak içinde)')
    parser.add_argument('--aggression', type=float, default=0.6,
                       help='Dönüşüm agresifliği (0.1-0.9)')
    
    args = parser.parse_args()
    
    # Yasal uyarıyı göster
    display_legal_warning()
    
    if args.demo:
        # Demonstrasyon modu
        results = demonstrate_system()
        print("\n" + "=" * 70)
        print("DEMONSTRASYON TAMAMLANDI")
        print("=" * 70)
        print("\nÖĞRENME ÇIKTILARI:")
        print("1. Metin dönüşüm tekniklerinin sınırlamalarını anladınız")
        print("2. Dedeksiyon sistemlerinin basit kurallarını gördünüz")
        print("3. Etik kullanımın önemini kavradınız")
        print("\nSONRAKİ ADIMLAR:")
        print("- Kendi metinlerinizle deney yapın (yalnızca kendi yazdıklarınızla)")
        print("- NLP kütüphanelerini (spaCy, NLTK) daha derinlemesine öğrenin")
        print("- Akademik yazım becerilerinizi geliştirin")
        
    elif args.text:
        # Özel metin analizi
        print(f"[*] Metin analizi başlatılıyor ({len(args.text.split())} kelime)...")
        
        transformer = AdvancedTextTransformer(aggression_level=args.aggression)
        result = transformer.execute_full_pipeline(args.text)
        
        tester = DetectionTester()
        analysis = tester.analyze_text(result.transformed_text)
        
        auditor = EthicalAuditor()
        ethics_report = auditor.audit_usage(
            original_text=args.text,
            transformed_text=result.transformed_text,
            techniques_used=result.techniques_used,
            context="user_analysis"
        )
        
        print("\n" + "=" * 70)
        print("SONUÇLAR:")
        print("=" * 70)
        print(f"Orijinal kelime: {len(args.text.split())}")
        print(f"Dönüştürülmüş kelime: {len(result.transformed_text.split())}")
        print(f"AI Skoru: {analysis['ai_score']:.3f}")
        print(f"İntihal Skoru: {analysis['plagiarism_score']:.3f}")
        print(f"Risk Seviyesi: {ethics_report['risk_level']}")
        print(f"Etik Öneri: {ethics_report['recommendation']}")
        
        if ethics_report['risk_level'] in ['HIGH', 'CRITICAL']:
            print("\n⚠️  YÜKSEK RİSK UYARISI:")
            for warning in ethics_report['specific_warnings']:
                print(f"   {warning}")
        
        print("\n" + "=" * 70)
        print("DÖNÜŞTÜRÜLMÜŞ METİN (ilk 500 karakter):")
        print("=" * 70)
        print(result.transformed_text[:500] + ("..." if len(result.transformed_text) > 500 else ""))
        
    else:
        # Etkileşimli mod
        print("Etkileşimli Akademik Analiz Modu")
        print("-" * 40)
        user_text = input("Analiz etmek istediğiniz metni girin (bitince boş satır bırakın):\n")
        
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        
        full_text = user_text + "\n" + "\n".join(lines)
        
        print(f"\n[*] {len(full_text.split())} kelime analiz ediliyor...")
        
        # Kısa analiz göster
        transformer = AdvancedTextTransformer(aggression_level=0.5)
        result = transformer.execute_full_pipeline(full_text[:500])  # İlk 500 kelime
        
        print("\nHızlı Analiz Sonucu:")
        print(f"Kullanılan teknikler: {', '.join(result.techniques_used[:3])}...")
        print(f"Risk skoru: {result.risk_score:.2f}")
        print(f"\nÖrnek dönüşüm (ilk 200 karakter):")
        print(result.transformed_text[:200] + "...")
        
        print("\n" + "=" * 70)
        print("ÖNEMLİ HATIRLATMA:")
        print("Bu araç yalnızca metin analizi tekniklerini ÖĞRENMEK içindir.")
        print("Gerçek akademik çalışmalar daima özgün olmalıdır.")
        print("=" * 70)