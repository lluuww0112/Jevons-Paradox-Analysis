import csv
import os

def is_empty(value):
    """값이 비어있는지 확인"""
    if value is None:
        return True
    if isinstance(value, (int, float)):
        return False
    return value == '' or str(value).strip() == ''

def convert_to_millions(value_str, unit):
    """값을 Million 단위로 변환"""
    if is_empty(value_str):
        return None, None
    
    try:
        value = float(value_str)
    except (ValueError, TypeError):
        return None, None
    
    if unit == 'B':
        return value * 1000, 'M'  # Billion to Million
    elif unit == 'M':
        return value, 'M'
    elif unit == 'K':
        return value / 1000, 'M'  # Thousand to Million
    else:
        # 단위가 없거나 알 수 없는 경우, 값의 크기로 추정
        if value >= 1000000:
            return value / 1000000, 'M'  # 단순 파라미터 수를 Million으로 변환
        elif value >= 1000:
            return value / 1000000, 'M'  # Thousand도 Million으로 변환
        else:
            return value, 'M'

def fix_csv_file(file_path):
    """CSV 파일을 수정"""
    print(f"\n처리 중: {file_path}")
    
    changes_made = []
    rows = []
    
    # CSV 파일 읽기
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
    
    # 각 행을 순회하며 수정
    for idx, row in enumerate(rows):
        backbone_count = row.get('backbone_parameter_count', '')
        backbone_unit = row.get('backbone_parameter_unit', '')
        learnable_count = row.get('learnable_parameter_count', '')
        learnable_unit = row.get('learnable_parameter_unit', '')
        learnable_ratio = row.get('learnable_parameter_ratio', '')
        
        # 백본 파라미터를 Million 단위로 변환
        if not is_empty(backbone_count) and not is_empty(backbone_unit):
            backbone_m, backbone_unit_m = convert_to_millions(backbone_count, backbone_unit)
        else:
            backbone_m = None
        
        # Learnable parameter count를 Million 단위로 변환 (의심스러운 값 확인)
        if not is_empty(learnable_count):
            try:
                learnable_count_val = float(learnable_count)
            except (ValueError, TypeError):
                learnable_count_val = None
            
            if learnable_count_val is not None:
                learnable_unit_val = learnable_unit if not is_empty(learnable_unit) else ''
                
                # 백본과 비교하여 의심스러운 값 확인
                # learnable이 backbone보다 크거나 비슷한 경우 (단위 오류로 추정)
                is_suspicious = False
                if backbone_m is not None:
                    # 단위가 M인데 learnable이 backbone보다 크거나 비슷한 경우
                    if learnable_unit_val == 'M' and learnable_count_val >= backbone_m * 0.5:
                        is_suspicious = True
                    # 단위가 M인데 값이 매우 큰 경우 (예: 1000 이상)
                    elif learnable_unit_val == 'M' and learnable_count_val > 1000:
                        is_suspicious = True
                    # 단위가 없는데 값이 큰 경우
                    elif learnable_unit_val == '' and learnable_count_val > 1000:
                        is_suspicious = True
                
                if is_suspicious:
                    # 단순 파라미터 수로 추정하여 Million으로 변환
                    learnable_m = learnable_count_val / 1000000
                    row['learnable_parameter_count'] = f"{learnable_m:.6f}"
                    row['learnable_parameter_unit'] = 'M'
                    changes_made.append(f"Row {idx+2}: learnable_parameter_count {learnable_count_val} -> {learnable_m:.6f}M (단위 수정: 단순 파라미터 수로 추정)")
                else:
                    learnable_m, learnable_unit_m = convert_to_millions(learnable_count_val, learnable_unit_val)
                    if learnable_m is not None:
                        if learnable_unit_m != learnable_unit_val and learnable_unit_val != '':
                            row['learnable_parameter_count'] = f"{learnable_m:.6f}"
                            row['learnable_parameter_unit'] = learnable_unit_m
                            changes_made.append(f"Row {idx+2}: learnable_parameter 단위 변환 {learnable_unit_val} -> {learnable_unit_m}")
                        elif learnable_unit_val == '':
                            row['learnable_parameter_count'] = f"{learnable_m:.6f}"
                            row['learnable_parameter_unit'] = learnable_unit_m
                            changes_made.append(f"Row {idx+2}: learnable_parameter 단위 추가 {learnable_unit_m}")
                        else:
                            learnable_m = learnable_count_val if learnable_unit_val == 'M' else None
                    else:
                        learnable_m = None
            else:
                learnable_m = None
        else:
            learnable_m = None
        
        # 역산으로 결측치 채우기
        if backbone_m is not None and backbone_m > 0:
            # learnable_ratio가 있고 learnable_count가 없는 경우
            if not is_empty(learnable_ratio) and is_empty(learnable_count):
                try:
                    ratio_val = float(learnable_ratio)
                    calculated_count = backbone_m * ratio_val
                    row['learnable_parameter_count'] = f"{calculated_count:.6f}"
                    row['learnable_parameter_unit'] = 'M'
                    changes_made.append(f"Row {idx+2}: learnable_parameter_count 계산됨 ({calculated_count:.6f}M) from ratio {learnable_ratio}")
                except (ValueError, TypeError):
                    pass
            
            # learnable_count가 있고 learnable_ratio가 없는 경우
            elif learnable_m is not None and is_empty(learnable_ratio):
                calculated_ratio = learnable_m / backbone_m
                row['learnable_parameter_ratio'] = f"{calculated_ratio:.6f}"
                changes_made.append(f"Row {idx+2}: learnable_parameter_ratio 계산됨 ({calculated_ratio:.6f}) from count {learnable_m}M")
            
            # 둘 다 있는 경우 일관성 확인
            elif learnable_m is not None and not is_empty(learnable_ratio):
                try:
                    ratio_val = float(learnable_ratio)
                    expected_ratio = learnable_m / backbone_m
                    if abs(expected_ratio - ratio_val) > 0.0001:  # 허용 오차
                        # ratio를 재계산하여 업데이트
                        row['learnable_parameter_ratio'] = f"{expected_ratio:.6f}"
                        changes_made.append(f"Row {idx+2}: learnable_parameter_ratio 수정됨 ({learnable_ratio} -> {expected_ratio:.6f})")
                except (ValueError, TypeError):
                    pass
    
    if changes_made:
        print(f"  총 {len(changes_made)}개의 변경사항:")
        for change in changes_made[:10]:  # 처음 10개만 출력
            print(f"    - {change}")
        if len(changes_made) > 10:
            print(f"    ... 외 {len(changes_made) - 10}개")
        
        # 파일 저장
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  파일 저장 완료: {file_path}")
    else:
        print(f"  변경사항 없음")
    
    return len(changes_made)

def main():
    base_dir = "/Users/yunsehyeog/Desktop/대학교/01_비교과/01_2025데이터톤/00_code/04_analysis/res_peft"
    
    years = ['2022', '2023', '2024', '2025']
    total_changes = 0
    
    for year in years:
        csv_path = os.path.join(base_dir, year, 'models.csv')
        if os.path.exists(csv_path):
            changes = fix_csv_file(csv_path)
            total_changes += changes
        else:
            print(f"\n파일을 찾을 수 없음: {csv_path}")
    
    print(f"\n\n총 {total_changes}개의 변경사항이 적용되었습니다.")

if __name__ == "__main__":
    main()
