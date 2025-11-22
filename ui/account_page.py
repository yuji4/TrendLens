import streamlit as st
import os
import shutil
from auth.auth_manager import delete_user


def render_account_page(username, user_dir):
    st.title("👤 내 계정 관리")

    st.markdown(f"현재 로그인 계정: **{username}**")
    st.caption("여기에서 데이터 파일을 관리하거나 계정 설정을 변경할 수 있습니다.")

    # ===========================
    # 📂 저장된 데이터 파일 목록
    # ===========================
    st.subheader("📂 저장된 데이터 파일 목록")

    if os.path.exists(user_dir):
        files = sorted(os.listdir(user_dir))

        if files:
            csv_files = [f for f in files if f.endswith(".csv")]

            if csv_files:
                st.write(f"총 **{len(csv_files)}개** CSV 파일이 저장되어 있습니다:")

                for f in csv_files:
                    st.write(f"📄 {f}")
            else:
                st.write("CSV 파일이 없습니다.")
        else:
            st.write("저장된 데이터가 없습니다.")
    else:
        st.error("⚠ 사용자 데이터 폴더가 존재하지 않습니다.")

    st.divider()

    # ===========================
    # 🗑 데이터 관리
    # ===========================
    st.subheader("🗑 데이터 관리")

    if st.button("❌ 모든 CSV 데이터 삭제", use_container_width=True):
        try:
            shutil.rmtree(user_dir, ignore_errors=True)
            os.makedirs(user_dir, exist_ok=True)

            st.success("모든 CSV 데이터가 삭제되었습니다.")
            st.rerun()
        except Exception as e:
            st.error(f"삭제 중 오류: {e}")

    st.divider()

    # ===========================
    # ⚠ 계정 설정
    # ===========================
    st.subheader("⚠ 계정 설정")

    logout_col, delete_col = st.columns(2)

    # 🚪 로그아웃
    with logout_col:
        if st.button("🚪 로그아웃", use_container_width=True):
            st.session_state.clear()
            st.session_state["logged_in"] = False
            st.rerun()

    # 🗑 회원탈퇴
    with delete_col:
        if st.button("🗑 회원탈퇴", type="primary", use_container_width=True):
            try:
                delete_user(username)
                shutil.rmtree(user_dir, ignore_errors=True)
                st.session_state.clear()
                st.success("회원탈퇴가 완료되었습니다.")
                st.rerun()
            except Exception as e:
                st.error(f"회원탈퇴 중 오류: {e}")

    st.divider()
    if st.button("⬅ 메인 페이지로 돌아가기", use_container_width=True):
        st.session_state["page"] = "main"
        st.rerun()
