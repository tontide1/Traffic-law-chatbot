import json
import os

benchmark_file = 'data/legal_benchmark.json'

new_scenarios = [
    {
        "id": "road-safety-corridor-retaining-wall",
        "question": "Hành lang an toàn đường bộ đối với đường bộ có kè, tường chắn bảo vệ nằm trong phạm vi đất dành cho kết cấu hạ tầng đường bộ được xác định như thế nào?",
        "category": "exactness",
        "query_class": "direct_definition",
        "expected_sources": ["Luật Đường bộ"],
        "required_terms": ["xác định từ mép ngoài của kè, tường chắn bảo vệ trở ra", "không lớn hơn chiều rộng hành lang an toàn đường bộ quy định tại các điểm a, b, c và d"],
        "forbidden_paraphrases": ["tính từ mép đường"],
        "forbidden_sources": [],
        "expected_answer_points": ["cách xác định đối với đường có kè, tường chắn"]
    },
    {
        "id": "billboard-construction-infrastructure",
        "question": "Việc xây dựng, lắp đặt biển quảng cáo trong phạm vi bảo vệ kết cấu hạ tầng đường bộ phải đáp ứng các yêu cầu gì để không ảnh hưởng an toàn giao thông?",
        "category": "exactness",
        "query_class": "direct_definition",
        "expected_sources": ["Luật Đường bộ"],
        "required_terms": ["Không được che khuất báo hiệu đường bộ", "không ảnh hưởng đến tầm nhìn", "cơ quan quản lý đường bộ chấp thuận bằng văn bản"],
        "forbidden_paraphrases": ["có thể tự do xây dựng"],
        "forbidden_sources": [],
        "expected_answer_points": ["yêu cầu an toàn", "sự chấp thuận của cơ quan quản lý"]
    },
    {
        "id": "urban-traffic-land-ratio",
        "question": "Tỷ lệ đất dành cho giao thông trên đất xây dựng đô thị được Luật Đường bộ quy định đạt từ bao nhiêu phần trăm?",
        "category": "exactness",
        "query_class": "direct_definition",
        "expected_sources": ["Luật Đường bộ"],
        "required_terms": ["từ 11% đến 26%", "đô thị có yếu tố đặc thù", "tối thiểu đạt 50% tỷ lệ đất quy định"],
        "forbidden_paraphrases": ["10%", "20%"],
        "forbidden_sources": [],
        "expected_answer_points": ["tỷ lệ phần trăm cho giao thông đô thị"]
    },
    {
        "id": "road-classification-management",
        "question": "Theo cấp quản lý, đường bộ được phân thành những loại đường nào?",
        "category": "exactness",
        "query_class": "direct_definition",
        "expected_sources": ["Luật Đường bộ"],
        "required_terms": ["quốc lộ", "đường tỉnh", "đường huyện", "đường xã", "đường thôn", "đường đô thị", "đường chuyên dùng"],
        "forbidden_paraphrases": ["đường liên ấp"],
        "forbidden_sources": [],
        "expected_answer_points": ["7 loại đường theo cấp quản lý"]
    },
    {
        "id": "expressway-toll-collection-state",
        "question": "Nhà nước thu phí sử dụng đường cao tốc đối với phương tiện lưu thông trên đường cao tốc nào do Nhà nước đại diện chủ sở hữu?",
        "category": "exactness",
        "query_class": "direct_definition",
        "expected_sources": ["Luật Đường bộ"],
        "required_terms": ["đường cao tốc do Nhà nước đầu tư theo hình thức đầu tư công", "đường cao tốc được đầu tư theo các hình thức khác khi kết thúc hợp đồng, chuyển giao cho Nhà nước"],
        "forbidden_paraphrases": ["mọi tuyến đường cao tốc"],
        "forbidden_sources": [],
        "expected_answer_points": ["phân loại đường cao tốc thu phí"]
    },
    {
        "id": "expressway-suspend-operation",
        "question": "Những trường hợp nào đường cao tốc phải tạm dừng khai thác?",
        "category": "graph_strength",
        "query_class": "relational",
        "expected_sources": ["Luật Đường bộ"],
        "required_terms": ["công trình bị hư hỏng do xảy ra sự cố", "xảy ra sự cố cháy, nổ, tai nạn giao thông đặc biệt nghiêm trọng", "yêu cầu phục vụ quốc phòng, an ninh"],
        "forbidden_paraphrases": ["khi trời mưa nhỏ"],
        "forbidden_sources": [],
        "expected_answer_points": ["lý do sự cố công trình", "tai nạn cháy nổ", "quốc phòng an ninh"]
    },
    {
        "id": "financial-sources-road-infrastructure",
        "question": "Nguồn thu từ kết cấu hạ tầng đường bộ nộp ngân sách nhà nước bao gồm những khoản phí nào?",
        "category": "exactness",
        "query_class": "direct_definition",
        "expected_sources": ["Luật Đường bộ"],
        "required_terms": ["Phí sử dụng đường bộ thu qua đầu phương tiện đối với xe ô tô", "Phí sử dụng đường cao tốc", "Nguồn thu liên quan đến khai thác, sử dụng kết cấu hạ tầng"],
        "forbidden_paraphrases": ["phí đỗ xe vỉa hè"],
        "forbidden_sources": [],
        "expected_answer_points": ["phí sử dụng đường bộ", "phí cao tốc"]
    },
    {
        "id": "school-bus-operation-rules",
        "question": "Hoạt động vận tải đưa đón trẻ em mầm non, học sinh bằng xe ô tô phải tuân thủ các quy định nào về loại hình vận tải?",
        "category": "exactness",
        "query_class": "direct_definition",
        "expected_sources": ["Luật Đường bộ"],
        "required_terms": ["cơ sở giáo dục tự tổ chức", "hoạt động vận tải nội bộ", "đơn vị kinh doanh vận tải thực hiện", "hoạt động kinh doanh vận tải"],
        "forbidden_paraphrases": ["ai chở cũng được"],
        "forbidden_sources": [],
        "expected_answer_points": ["vận tải nội bộ nếu tự tổ chức", "kinh doanh vận tải nếu thuê ngoài"]
    },
    {
        "id": "taxi-fare-calculation-methods",
        "question": "Tiền cước chuyến đi đối với kinh doanh vận tải hành khách bằng xe taxi do hành khách lựa chọn theo những phương thức nào?",
        "category": "exactness",
        "query_class": "direct_definition",
        "expected_sources": ["Luật Đường bộ"],
        "required_terms": ["đồng hồ tính tiền", "phần mềm tính tiền có kết nối trực tiếp", "thoả thuận với đơn vị kinh doanh vận tải"],
        "forbidden_paraphrases": ["chỉ dùng đồng hồ"],
        "forbidden_sources": [],
        "expected_answer_points": ["3 phương thức tính cước taxi"]
    },
    {
        "id": "car-rental-service-requirements",
        "question": "Dịch vụ cho thuê phương tiện giao thông cơ giới đường bộ để tự lái đối với xe ô tô yêu cầu gì về giấy phép lái xe của người thuê?",
        "category": "exactness",
        "query_class": "direct_definition",
        "expected_sources": ["Luật Đường bộ"],
        "required_terms": ["có giấy phép lái xe đang còn điểm", "còn hiệu lực phù hợp với loại xe cho thuê", "bản phô tô giấy phép lái xe"],
        "forbidden_paraphrases": ["không cần bằng lái"],
        "forbidden_sources": [],
        "expected_answer_points": ["yêu cầu bằng lái còn điểm", "hợp đồng kèm bản phô tô"]
    },
    {
        "id": "child-seat-car-rules",
        "question": "Luật Trật tự, an toàn giao thông đường bộ quy định như thế nào về việc chở trẻ em dưới 10 tuổi và chiều cao dưới 1,35 mét trên xe ô tô?",
        "category": "exactness",
        "query_class": "direct_definition",
        "expected_sources": ["Luật Trật tự, an toàn giao thông đường bộ"],
        "required_terms": ["không được cho trẻ em ngồi cùng hàng ghế với người lái xe", "trừ loại xe ô tô chỉ có một hàng ghế", "phải sử dụng thiết bị an toàn phù hợp"],
        "forbidden_paraphrases": ["được ngồi ghế phụ nếu thắt dây"],
        "forbidden_sources": [],
        "expected_answer_points": ["cấm ngồi ghế phụ", "bắt buộc dùng thiết bị an toàn"]
    },
    {
        "id": "prohibited-acts-alcohol",
        "question": "Hành vi nào liên quan đến nồng độ cồn bị nghiêm cấm khi tham gia giao thông đường bộ?",
        "category": "exactness",
        "query_class": "direct_definition",
        "expected_sources": ["Luật Trật tự, an toàn giao thông đường bộ"],
        "required_terms": ["Điều khiển phương tiện tham gia giao thông đường bộ mà trong máu hoặc hơi thở có nồng độ cồn"],
        "forbidden_paraphrases": ["vượt quá 50 miligam mới cấm"],
        "forbidden_sources": [],
        "expected_answer_points": ["cấm tuyệt đối nồng độ cồn"]
    },
    {
        "id": "urban-horn-usage-time",
        "question": "Việc sử dụng còi trong khu đông dân cư, khu vực cơ sở khám bệnh, chữa bệnh bị cấm trong khoảng thời gian nào (trừ xe ưu tiên)?",
        "category": "exactness",
        "query_class": "direct_definition",
        "expected_sources": ["Luật Trật tự, an toàn giao thông đường bộ"],
        "required_terms": ["từ 22 giờ ngày hôm trước đến 05 giờ ngày hôm sau"],
        "forbidden_paraphrases": ["từ 23 giờ", "đến 6 giờ sáng"],
        "forbidden_sources": [],
        "expected_answer_points": ["khung giờ cấm bấm còi"]
    },
    {
        "id": "roundabout-right-of-way",
        "question": "Tại nơi đường giao nhau có báo hiệu đi theo vòng xuyến, người lái xe phải nhường đường như thế nào?",
        "category": "exactness",
        "query_class": "direct_definition",
        "expected_sources": ["Luật Trật tự, an toàn giao thông đường bộ"],
        "required_terms": ["nhường đường cho xe đi đến từ bên trái"],
        "forbidden_paraphrases": ["nhường bên phải"],
        "forbidden_sources": [],
        "expected_answer_points": ["nhường đường cho xe bên trái tại vòng xuyến"]
    },
    {
        "id": "ferry-pontoon-bridge-priority",
        "question": "Thứ tự ưu tiên đối với các xe khi qua phà, cầu phao được quy định từ trên xuống dưới như thế nào?",
        "category": "graph_strength",
        "query_class": "relational",
        "expected_sources": ["Luật Trật tự, an toàn giao thông đường bộ"],
        "required_terms": ["Xe ưu tiên", "Xe chở thư báo", "Xe chở thực phẩm tươi sống", "Xe chở khách công cộng"],
        "forbidden_paraphrases": ["xe tải qua trước"],
        "forbidden_sources": [],
        "expected_answer_points": ["thứ tự 4 loại xe ưu tiên qua phà"]
    },
    {
        "id": "passing-on-the-right-exceptions",
        "question": "Trong trường hợp nào thì người lái xe được phép vượt về bên phải xe phía trước?",
        "category": "graph_strength",
        "query_class": "relational",
        "expected_sources": ["Luật Trật tự, an toàn giao thông đường bộ"],
        "required_terms": ["xe phía trước có tín hiệu rẽ trái hoặc đang rẽ trái", "xe chuyên dùng đang làm việc trên đường mà không thể vượt bên trái"],
        "forbidden_paraphrases": ["vượt bên phải thoải mái"],
        "forbidden_sources": [],
        "expected_answer_points": ["2 trường hợp được vượt phải"]
    },
    {
        "id": "family-member-responsibility-seatbelt",
        "question": "Thành viên trong gia đình có trách nhiệm gì đối với việc sử dụng dây đai an toàn hoặc ghế dành cho trẻ em?",
        "category": "exactness",
        "query_class": "direct_definition",
        "expected_sources": ["Luật Trật tự, an toàn giao thông đường bộ"],
        "required_terms": ["tuyên truyền, nhắc nhở thành viên khác chấp hành", "sử dụng dây đai an toàn", "ghế dành cho trẻ em", "có người lớn ngồi cùng trẻ em phía sau khi chở trẻ em dưới 06 tuổi bằng xe gắn máy, xe mô tô"],
        "forbidden_paraphrases": ["không quan tâm"],
        "forbidden_sources": [],
        "expected_answer_points": ["trách nhiệm nhắc nhở thắt dây đai", "quy định chở trẻ dưới 6 tuổi bằng xe máy"]
    },
    {
        "id": "u-turn-prohibited-locations",
        "question": "Những vị trí nào cấm quay đầu xe theo Luật Trật tự, an toàn giao thông đường bộ?",
        "category": "graph_strength",
        "query_class": "relational",
        "expected_sources": ["Luật Trật tự, an toàn giao thông đường bộ"],
        "required_terms": ["phần đường dành cho người đi bộ qua đường", "trên cầu, đầu cầu, gầm cầu vượt, ngầm", "nơi đường bộ giao nhau cùng mức với đường sắt", "trong hầm đường bộ", "trên đường cao tốc", "đường một chiều"],
        "forbidden_paraphrases": [],
        "forbidden_sources": [],
        "expected_answer_points": ["liệt kê các vị trí cấm quay đầu"]
    },
    {
        "id": "reversing-vehicle-prohibited-locations",
        "question": "Không được lùi xe ở những khu vực nào?",
        "category": "graph_strength",
        "query_class": "relational",
        "expected_sources": ["Luật Trật tự, an toàn giao thông đường bộ"],
        "required_terms": ["đường một chiều", "khu vực cấm dừng", "phần đường dành cho người đi bộ qua đường", "nơi tầm nhìn bị che khuất", "trong hầm đường bộ", "trên đường cao tốc"],
        "forbidden_paraphrases": ["được lùi nếu có người xi nhan"],
        "forbidden_sources": [],
        "expected_answer_points": ["liệt kê các vị trí cấm lùi xe"]
    },
    {
        "id": "headlight-usage-time",
        "question": "Thời gian bắt buộc phải bật đèn chiếu sáng phía trước khi tham gia giao thông là từ mấy giờ đến mấy giờ?",
        "category": "exactness",
        "query_class": "direct_definition",
        "expected_sources": ["Luật Trật tự, an toàn giao thông đường bộ"],
        "required_terms": ["từ 18 giờ ngày hôm trước đến 06 giờ ngày hôm sau", "có sương mù, khói, bụi, trời mưa, thời tiết xấu làm hạn chế tầm nhìn"],
        "forbidden_paraphrases": ["từ 19 giờ đến 5 giờ sáng"],
        "forbidden_sources": [],
        "expected_answer_points": ["khung giờ bật đèn chiếu sáng theo luật mới"]
    }
]

with open(benchmark_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

data.extend(new_scenarios)

with open(benchmark_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Successfully appended {len(new_scenarios)} new scenarios. Total is now {len(data)}.")
