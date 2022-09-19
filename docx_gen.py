from __future__ import print_function
from mailmerge import MailMerge
from datetime import date
from docxtpl import DocxTemplate
import jinja2

# https://pbpython.com/python-word-template.html

# template = "data/TEMPLATE.docx"
# with  MailMerge(template) as document:
#     print(document.get_merge_fields())
#     document.merge(
#     nip_prakom="198401112009011004",
#     nip_atasan="198401112009011005",
#     kode_kegiatan="III.B.1",
#     bukti_kegiatan="1) a, \n\n\n2) b\n\n\n, 3) c\n\n\n",
#     judul_kegiatan="pembuatan model AI",
#     lokasi="karlsruhe",
#     query_kegiatan="buat model machine learning",
#     keterangan_kegiatan="-",
#     nama_atasan="ismail fahmi",
#     nama_prakom="redy andri",
#     pangkat="Penaa Tk. I",
#     jenjang_prakom="ahli muda",
#     tanggal="23-agustus-2022",
#     angka_kredit="1,25",
#     golongan="IIIB"
#     )
#     document.write('test-output.docx')



# print(doc.getMergeFields())
# context = {
# "nip_prakom":"198401112009011004",
# "nip_atasan":"198401112009011005",
# "kode_kegiatan":"III.B.1",
# "bukti_kegiatan":"1) a, \n\n\n2) b\n\n\n, 3) c\n\n\n",
# "judul_kegiatan":"pembuatan model AI",
# "lokasi":"karlsruhe",
# "query_kegiatan":"buat model machine learning",
# "keterangan_kegiatan":"-",
# "nama_atasan":"ismail fahmi",
# "nama_prakom":"redy andri",
# "pangkat":"Penaa Tk. I",
# "jenjang_prakom":"ahli muda",
# "tanggal":"23-agustus-2022",
# "angka_kredit":"1,25",
# "golongan":"IIIB"
# }
template = "data/TEMPLATE2.docx"
tpl = DocxTemplate(template)
context = {}
context["nip_prakom"]="198401112009011004"
context["nip_atasan"]="198401112009011005"
context["kode_kegiatan"]="III.B.1"
context["bukti_kegiatan"]="1) a, \n\n\n2) b\n\n\n, 3) c\n\n\n"
context["judul_kegiatan"]="pembuatan model AI"
context["lokasi"]="karlsruhe"
context["query_kegiatan"]="buat model machine learning"
context["keterangan_kegiatan"]="-"
context["nama_atasan"]="ismail fahmi"
context["nama_prakom"]="redy andri"
context["pangkat"]="Penaa Tk. I"
context["jenjang_prakom"]="ahli muda"
context["tanggal"]="23-agustus-2022"
context["angka_kredit"]="1,25"
context["golongan"]="IIIB"
tpl.render(context)
tpl.save("doctpl.docx")
from io import StringIO
file_stream = StringIO()
tpl.save(file_stream)

# return send_file(file_stream, as_attachment=True, attachment_filename='report_'+user_id+'.docx')