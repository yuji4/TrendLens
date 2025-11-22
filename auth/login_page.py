import streamlit as st
from auth.auth_manager import create_user, verify_user, init_db

init_db()

# -------------------------------
# 🔐 로그인 페이지
# -------------------------------
def show_login_page():
    st.title("🔐 TrendLens 로그인")
    st.markdown("로그인 후 검색 트렌드 분석 기능을 이용할 수 있습니다.")

    username = st.text_input(
        "아이디",
        placeholder="등록한 아이디를 입력하세요",
        key="login_username"
    )
    password = st.text_input(
        "비밀번호",
        type="password",
        placeholder="비밀번호를 입력하세요",
        key="login_password"
    )

    if st.button("로그인"):
        if verify_user(username, password):
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.success(f"환영합니다, {username}님!")
            st.rerun()
        else:
            st.error("아이디 또는 비밀번호가 올바르지 않습니다.")


# -------------------------------
# 📝 회원가입 페이지
# -------------------------------
def show_signup_page():
    st.title("📝 TrendLens 회원가입")
    st.markdown("새 계정을 만들어주세요.")

    username = st.text_input(
        "아이디",
        placeholder="영문/숫자 조합 4~20자",
        key="signup_username"
    )
    password = st.text_input(
        "비밀번호",
        type="password",
        placeholder="8자 이상, 특수문자/숫자 포함 권장",
        key="signup_password"
    )
    password_confirm = st.text_input(
        "비밀번호 확인",
        type="password",
        placeholder="비밀번호를 다시 입력하세요",
        key="signup_password_confirm"
    )

    if st.button("회원가입"):
        # 입력 검증
        if not username or not password:
            st.error("아이디와 비밀번호를 모두 입력해주세요.")
        elif password != password_confirm:
            st.error("비밀번호가 일치하지 않습니다.")
        else:
            created = create_user(username, password)
            if created:
                st.success("회원가입 완료! 이제 로그인해주세요.")

                # 🔥 회원가입 직후 자동 로그인 화면으로 이동
                st.session_state["signup_mode"] = False
                st.rerun()
            else:
                st.error("이미 존재하는 아이디입니다. 다른 아이디를 사용해주세요.")


# -------------------------------
# 🔄 로그인/회원가입 화면 전환 버튼
# -------------------------------
def show_auth_switch():
    if "signup_mode" not in st.session_state:
        st.session_state["signup_mode"] = False

    if st.session_state["signup_mode"]:
        if st.button("← 로그인 페이지로 돌아가기"):
            st.session_state["signup_mode"] = False
            st.rerun()
    else:
        if st.button("회원가입 하기"):
            st.session_state["signup_mode"] = True
            st.rerun()


# -------------------------------
# 🔑 인증 메인 컨트롤러
# -------------------------------
def render_auth_page():
    st.markdown("<style>footer{visibility:hidden;}</style>", unsafe_allow_html=True)

    # 회원가입 모드면 signup 페이지 보여주기
    if st.session_state.get("signup_mode", False):
        show_signup_page()
    else:
        show_login_page()

    st.divider()
    show_auth_switch()

    return st.session_state.get("logged_in", False)
