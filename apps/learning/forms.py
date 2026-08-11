from io import BytesIO

from django import forms
from django.contrib.auth import authenticate, get_user_model
from django.contrib.auth.forms import UserCreationForm
from PIL import Image, UnidentifiedImageError

from .models import UserProfile


User = get_user_model()


class SignupForm(UserCreationForm):
    display_name = forms.CharField(label="Nama lengkap", max_length=120)
    email = forms.EmailField(label="Email")

    class Meta:
        model = User
        fields = ("display_name", "email", "password1", "password2")

    def clean_email(self):
        email = self.cleaned_data["email"].lower()
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError("Email sudah terdaftar.")
        return email

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data["email"].lower()
        user.username = self.cleaned_data["email"].lower()
        user.display_name = self.cleaned_data["display_name"]
        if commit:
            user.save()
        return user


class LoginForm(forms.Form):
    email = forms.EmailField(label="Email")
    password = forms.CharField(label="Kata sandi", widget=forms.PasswordInput)

    def __init__(self, request=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.request = request
        self.user = None
        self._apply_control_classes()

    def _apply_control_classes(self):
        control_class = "w-full rounded border border-slate-300 bg-white px-3 py-3 text-sm text-slate-900 shadow-sm focus:border-[var(--color-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary-tint)]"
        for field in self.fields.values():
            existing = field.widget.attrs.get("class", "")
            field.widget.attrs["class"] = f"{existing} {control_class}".strip()

    def clean(self):
        cleaned = super().clean()
        email = cleaned.get("email")
        password = cleaned.get("password")
        if email and password:
            self.user = authenticate(self.request, username=email.lower(), password=password)
            if self.user is None:
                raise forms.ValidationError("Email atau kata sandi tidak sesuai.")
        return cleaned


class ProfileForm(forms.ModelForm):
    display_name = forms.CharField(label="Nama", max_length=120)
    remove_photo = forms.BooleanField(label="Hapus foto profil", required=False)

    class Meta:
        model = UserProfile
        fields = ("display_name", "profile_photo", "bio", "learning_level", "remove_photo")
        labels = {
            "profile_photo": "Foto profil",
            "bio": "Bio singkat",
            "learning_level": "Level BISINDO",
        }

    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop("user")
        super().__init__(*args, **kwargs)
        self.fields["display_name"].initial = self.user.display_name or self.user.get_full_name()
        self._apply_control_classes()

    def _apply_control_classes(self):
        control_class = "w-full rounded border border-slate-300 bg-white px-3 py-3 text-sm text-slate-900 shadow-sm focus:border-[var(--color-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary-tint)]"
        for field in self.fields.values():
            if isinstance(field.widget, forms.CheckboxInput):
                field.widget.attrs["class"] = "h-4 w-4 rounded border-slate-300 accent-[var(--color-primary)]"
                continue
            existing = field.widget.attrs.get("class", "")
            field.widget.attrs["class"] = f"{existing} {control_class}".strip()

    def clean_profile_photo(self):
        photo = self.cleaned_data.get("profile_photo")
        if not photo:
            return photo
        try:
            image = Image.open(photo)
            image.verify()
        except (UnidentifiedImageError, OSError):
            raise forms.ValidationError("File yang diunggah bukan gambar yang valid.")
        photo.seek(0)
        return photo

    def save(self, commit=True):
        profile = super().save(commit=False)
        self.user.display_name = self.cleaned_data["display_name"]
        if self.cleaned_data.get("remove_photo") and profile.profile_photo:
            profile.profile_photo.delete(save=False)
            profile.profile_photo = None
        if commit:
            self.user.save(update_fields=["display_name"])
            profile.save()
            self._resize_photo(profile)
        return profile

    def _resize_photo(self, profile):
        if not profile.profile_photo:
            return
        image = Image.open(profile.profile_photo.path)
        image.thumbnail((512, 512))
        output = BytesIO()
        image_format = "PNG" if image.mode in {"RGBA", "P"} else "JPEG"
        if image_format == "JPEG":
            image = image.convert("RGB")
        image.save(output, image_format, quality=85, optimize=True)
        with open(profile.profile_photo.path, "wb") as photo_file:
            photo_file.write(output.getvalue())


class OnboardingForm(forms.ModelForm):
    GOAL_CHOICES = [
        ("daily", "Komunikasi sehari-hari"),
        ("deaf_family_friends", "Berkomunikasi dengan keluarga/teman Tuli"),
        ("school", "Sekolah atau kampus"),
        ("work_service", "Dunia kerja / pelayanan"),
        ("self", "Belajar untuk diri sendiri"),
    ]
    FAMILIARITY_CHOICES = [
        ("new", "Belum pernah belajar"),
        ("some", "Pernah belajar sedikit"),
        ("basic", "Sudah memahami dasar"),
    ]

    learning_goal = forms.ChoiceField(label="Apa tujuanmu belajar BISINDO?", choices=GOAL_CHOICES, widget=forms.RadioSelect)
    bisindo_familiarity = forms.ChoiceField(label="Seberapa familiar kamu dengan BISINDO?", choices=FAMILIARITY_CHOICES, widget=forms.RadioSelect)

    class Meta:
        model = UserProfile
        fields = ("learning_goal", "bisindo_familiarity")

    def save(self, commit=True):
        profile = super().save(commit=False)
        profile.onboarding_completed = True
        if commit:
            profile.save(update_fields=["learning_goal", "bisindo_familiarity", "onboarding_completed", "updated_at"])
        return profile
