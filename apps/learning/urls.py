from django.urls import path

from . import views

urlpatterns = [
    path("", views.LandingPageView.as_view(), name="landing"),
    path("accounts/signup/", views.SignupView.as_view(), name="signup"),
    path("accounts/login/", views.LoginView.as_view(), name="login"),
    path("accounts/logout/", views.LogoutView.as_view(), name="logout"),
    path("accounts/password/reset/", views.PasswordResetView.as_view(), name="password_reset"),
    path("profil/", views.ProfileView.as_view(), name="profile"),
    path("profil/edit/", views.ProfileEditView.as_view(), name="profile_edit"),
    path("library/", views.LearningLibraryView.as_view(), name="library"),
    path("lessons/<slug:slug>/", views.LessonDetailView.as_view(), name="lesson_detail"),
    path("lessons/<slug:slug>/practice/", views.StartPracticeView.as_view(), name="start_practice"),
    path("practice/<int:session_id>/camera/", views.AIPracticeView.as_view(), name="practice_camera"),
    path("practice/<int:session_id>/analyze/", views.AnalyzePracticeView.as_view(), name="analyze_practice"),
    path("practice/<int:session_id>/result/", views.PracticeResultView.as_view(), name="practice_result"),
    path("dashboard/", views.ProgressDashboardView.as_view(), name="dashboard"),
]
